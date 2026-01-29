# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
"""
Simple CUDA IPC tensor transfer.

This is the most straightforward implementation for same-GPU tensor transfer
using CUDA IPC and environment variable configuration.

Usage:
    # Set environment variables
    export IPC_PATH=/tmp/ipc_socket

    # In producer process:
    producer = ProducerIpc(rank=0)
    producer.send_pic(tensors, "request_001")
    
    # In consumer process:
    consumer = ConsumerIpc(rank=0)
    tensors = consumer.receive_pic(num_layers=3, "request_001")
"""
import logging
import os
import socket
import struct
import threading
import time
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.reductions import reduce_tensor
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class SocketBase:
    """Managing a socket."""

    def __init__(self, rank: int, timeout: float = 10.0):
        self.rank = rank
        self.device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        
        self.max_retries = 5
        self.ipc_path = os.environ.get("IPC_PATH", "/tmp/ipc_socket")
        self.ipc_path += f"_{rank}"
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        
        logger.debug(f"Initialized {self.__class__.__name__} on GPU {rank}")
    
    def __del__(self):
        self.sock.close()
        if os.path.exists(self.ipc_path):
            os.remove(self.ipc_path)

    def _send_msg(self, sock: socket.socket, msg: Dict[str, Any]):
        """Send message with length prefix."""
        import pickle
        data = pickle.dumps(msg)
        sock.sendall(struct.pack('!I', len(data)) + data)
    
    def _recv_msg(self, sock: socket.socket) -> Dict[str, Any]:
        """Receive message with length prefix."""
        import pickle
        length_data = self._recv_all(sock, 4)
        length = struct.unpack('!I', length_data)[0]
        data = self._recv_all(sock, length)
        return pickle.loads(data)
    
    def _recv_all(self, sock: socket.socket, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection broken")
            data += chunk
        return data

class ProducerIpc(SocketBase):
    """Simple producer for CUDA IPC tensor transfer."""
    role = "producer"
    def __init__(self, rank: int, timeout: float = 120.0):
        # NOTE: We must set large timeout to wait for LLM engine initialization.
        super().__init__(rank, timeout)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if os.path.exists(self.ipc_path):
            os.remove(self.ipc_path)

        self.sock.bind(self.ipc_path)
        self.sock.listen(1)
        self.pic_handles: dict[str, List[Dict[str, Any]]] = {}
        self.stop_event = threading.Event()
        self.loop_thread = threading.Thread(target=self.loop)
        self.loop_thread.start()

    def close(self):
        """Gracefully shut down the producer."""
        if hasattr(self, 'stop_event') and not self.stop_event.is_set():
            logger.debug("Producer: Shutting down loop thread...")
            self.stop_event.set()
            self.loop_thread.join(timeout=10)
            if self.loop_thread.is_alive():
                logger.warning("Producer: Loop thread did not terminate gracefully")

    def __del__(self):
        # Still include for safety, but don't rely on it
        self.close()
        super().__del__()
    
    def send_pic(self, pic_tensors: List[Tuple[torch.Tensor, torch.Tensor]], request_id: str):
        """
        Send PIC tensors to consumer process.
        
        Args:
            pic_tensors: List of (key, value) tensor pairs
            request_id: Unique request identifier
            
        Returns:
            True if successful
        """
        shared_handles = []
        for i, (key, value) in enumerate(pic_tensors):
            key_handle = reduce_tensor(key)
            value_handle = reduce_tensor(value)
            shared_handles.append({
                'key_handle': key_handle,
                'value_handle': value_handle,
            })
            logger.debug(f"Layer {i}: {key.shape} + {value.shape}")

        self.pic_handles[request_id] = shared_handles
    
    def loop(self):
        logger.debug(f"Producer: Waiting for consumer on {self.ipc_path}")
        conn, addr = self.sock.accept()
        logger.debug(f"Producer: Consumer connected from {addr}")
        while not self.stop_event.is_set():
            try:
                request_id = conn.recv(32).decode()
                if not request_id:
                    # Shutdown
                    logger.info("Producer: Loop thread shutdown.")
                    break

                for attempt in range(self.max_retries):
                    handles = self.pic_handles.pop(request_id, None)
                    if handles is not None:
                        break
                    wait_time = 0.5 * (2 ** attempt)
                    logger.debug(f"Producer: Request ID not found, waiting {wait_time}s...")
                    time.sleep(wait_time)

                if handles is None:
                    logger.error(f"Producer: Request ID not found: {request_id}")
                    conn.sendall(b'ERROR')
                    continue

                self._send_msg(conn, handles)
                
                # Wait for completion
                signal = conn.recv(4)
                success = signal == b'DONE'
                
                logger.debug(f"Producer: Transfer {'completed' if success else 'failed'} for {request_id}")
                
            except socket.timeout:
                logger.debug("Producer: Timeout, relistening for new connection")
            except Exception as e:
                logger.error(f"Producer: Error sending {request_id}: {e}")

        conn.close()

class ConsumerIpc(SocketBase):
    """Simple consumer for CUDA IPC tensor transfer."""
    role = "consumer"
    def __init__(self, rank: int, timeout: float = 10.0):
        super().__init__(rank, timeout)

        # Connect to producer with retry
        for attempt in range(self.max_retries):
            try:
                self.sock.connect(self.ipc_path)
                break
            except ConnectionRefusedError:
                if attempt < self.max_retries - 1:
                    wait_time = 0.5 * (2 ** attempt)
                    logger.debug(f"Consumer: Producer not ready, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        logger.debug("Consumer: Connected to producer")

    def receive_pic(self, num_layers: int, request_id: str) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Receive PIC tensors from producer process.
        
        Args:
            num_layers: Expected number of tensor pairs
            request_id: Unique request identifier
            timeout: Timeout in seconds
            
        Returns:
            List of (key, value) tensor pairs or None
        """
        try:
            logger.debug(f"Consumer: Receiving for {request_id}")
            self.sock.sendall(request_id.encode())
            
            # Step 1: Receive tensor information
            handles = self._recv_msg(self.sock)
            if len(handles) != num_layers:
                logger.warning(f"Consumer: Expected {num_layers} layers, got {len(handles)}")
            
            # Step 2: Reconstruct tensors
            tensors = []
            for handle in handles:
                key_handle, value_handle = handle['key_handle'], handle['value_handle']
                rebuild_fn, args = key_handle
                key = rebuild_fn(*args)
                rebuild_fn, args = value_handle
                value = rebuild_fn(*args)
                tensors.append((key, value))
            
            # Step 3: Send completion signal
            self.sock.sendall(b'DONE')
            logger.debug(f"Consumer: Successfully received tensor for {request_id}")
            return tensors
            
        except Exception as e:
            logger.error(f"Consumer: Error receiving {request_id}: {e}")
            return None

def run_simple_example():
    """Run a simple example demonstrating the tensor transfer."""
    
    def producer_worker(rank: int):
        """Producer worker process."""
        print(f"\n=== Producer (GPU {rank}) ===")
        
        producer = ProducerIpc(rank)
        
        # Create sample tensors
        print("Creating sample tensors...")
        pic_tensors = []
        for i in range(2):  # 2 layers
            batch, seq_len, dim = 4, 64, 384
            key = torch.randn(batch, seq_len, dim, dtype=torch.float16, device=f'cuda:{rank}')
            value = torch.randn(batch, seq_len, dim, dtype=torch.float16, device=f'cuda:{rank}')
            pic_tensors.append((key, value))
            print(f"  Layer {i}: key={key.shape}, value={value.shape}")
        
        # Send tensors
        request_id = "simple_test_001"
        print(f"\nSending request: {request_id}")
        producer.send_pic(pic_tensors, request_id)
        time.sleep(5)
        producer.close()
    
    def consumer_worker(rank: int):
        """Consumer worker process."""
        print(f"\n=== Consumer (GPU {rank}) ===")
        
        # Small delay
        time.sleep(2)
        
        consumer = ConsumerIpc(rank)
        
        # Receive tensors
        request_id = "simple_test_001"
        print(f"Receiving request: {request_id}")
        
        tensors = consumer.receive_pic(num_layers=2, request_id=request_id)
        
        if tensors:
            print(f"\n✓ Received {len(tensors)} tensor pairs:")
            for i, (key, value) in enumerate(tensors):
                print(f"  Layer {i}:")
                print(f"    Key:   {key.shape} @ {key.device} (dtype: {key.dtype})")
                print(f"    Value: {value.shape} @ {value.device} (dtype: {value.dtype})")
                
                # Verify zero-copy: tensors should be on same device
                assert str(key.device) == f'cuda:{rank}'
                assert str(value.device) == f'cuda:{rank}'
        else:
            print("\n✗ Receive failed")
    
    # Run on GPU 0
    rank = 0
    
    # Start producer
    producer_proc = mp.Process(target=producer_worker, args=(rank,))
    producer_proc.start()

    # Start consumer
    consumer_proc = mp.Process(target=consumer_worker, args=(rank,))
    consumer_proc.start()
    
    # Wait for completion
    producer_proc.join()
    consumer_proc.join()
    
    print("\n=== Simple example completed ===")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    run_simple_example()
