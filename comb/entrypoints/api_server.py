# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
"""
Async HTTP API Server for COMB System.

This module provides an async API server for serving COMB inference requests.
It supports both streaming and non-streaming response modes.
"""

import asyncio
import json
import logging
from argparse import ArgumentParser, Namespace
from collections.abc import AsyncGenerator
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from comb.entrypoints.comb import COMB
from comb.output import RequestOutput

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI()
async_comb_engine = None


class AsyncCOMB:
    """Async wrapper for COMB to handle requests asynchronously.
    
    This class wraps the synchronous COMB engine to provide async-compatible
    interfaces for HTTP request handling.
    """
    
    def __init__(
        self,
        model: str,
        num_instances: int = 1,
        pic_memory_utilization: float = 0.3,
        pbc_memory_utilization: float = 0.5,
        pic_separated: bool = False,
        **kwargs,
    ):
        """Initialize AsyncCOMB engine.
        
        Args:
            model: The name or path of a COMB model.
            num_instances: The number of inference engines.
            pic_memory_utilization: The ratio of GPU memory for PIC (0-1).
            pbc_memory_utilization: The ratio of GPU memory for prefix-based cache (0-1).
            pic_separated: Whether to assign different GPUs for PIC and LLM engine.
            **kwargs: Additional arguments passed to COMB.
        """
        self.comb = COMB(
            model=model,
            num_instances=num_instances,
            pic_memory_utilization=pic_memory_utilization,
            pbc_memory_utilization=pbc_memory_utilization,
            pic_separated=pic_separated,
            **kwargs,
        )
        self.model = model
        
    async def generate(
        self,
        prompt: dict[str, Any],
        **kwargs,
    ) -> Any:
        """Generate output asynchronously.
        
        This method runs the synchronous COMB.generate() in a thread pool
        to avoid blocking the async event loop.
        
        Args:
            prompt: The input prompt dictionary.
            **kwargs: Additional generation parameters.
            
        Returns:
            The generated output from COMB.
        """
        loop = asyncio.get_event_loop()
        
        # Run the synchronous generate method in a thread pool
        result = await loop.run_in_executor(
            None,
            lambda: self.comb.generate_for_single_request(prompt, **kwargs)
        )
        return result


@app.get("/health")
async def health() -> Response:
    """Health check endpoint."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> RequestOutput:
    """Generate completion for the request.
    
    The request should be a JSON object with the following fields:
    - prompt: the prompt dictionary for COMB generation
    - stream: whether to stream the results or not (default: False)
    - other fields: additional generation parameters
    """
    request_dict = await request.json()
    return await _generate(request_dict)


async def _generate(request_dict: dict) -> RequestOutput:
    """Internal generate handler with cancellation support."""
    try:
        prompt = request_dict.pop("prompt")
        stream = request_dict.pop("stream", False)
        request_id = request_dict.pop("request_id", None)
        need_store = request_dict.pop("need_store", True)
        
        # Remaining items are additional generation parameters
        gen_params = request_dict
        
        assert async_comb_engine is not None
        
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            try:
                result = await async_comb_engine.generate(
                    prompt, 
                    need_store=need_store,
                    **gen_params
                )
                yield (json.dumps(result) + "\n").encode("utf-8")
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                error_ret = {
                    "request_id": request_id,
                    "error": str(e),
                }
                yield (json.dumps(error_ret) + "\n").encode("utf-8")
        
        if stream:
            return StreamingResponse(stream_results())
        
        # Non-streaming case
        try:
            result = await async_comb_engine.generate(
                prompt,
                need_store=need_store,
                **gen_params
            )
            return result
        except asyncio.CancelledError:
            return Response(status_code=499)
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            ret = {
                "request_id": request_id,
                "error": str(e),
            }
            return JSONResponse(ret, status_code=500)
            
    except KeyError as e:
        logger.error(f"Missing required field: {e}")
        return JSONResponse(
            {"error": f"Missing required field: {e}"},
            status_code=400
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )


def build_app(args: Namespace) -> FastAPI:
    """Build and configure FastAPI app."""
    global app
    
    app.root_path = getattr(args, "root_path", "")
    return app


async def init_app(
    args: Namespace,
    comb_engine: Optional[AsyncCOMB] = None,
) -> FastAPI:
    """Initialize the FastAPI app with COMB engine."""
    app_instance = build_app(args)
    
    global async_comb_engine
    
    if comb_engine is not None:
        async_comb_engine = comb_engine
    else:
        # Create AsyncCOMB engine from arguments
        async_comb_engine = AsyncCOMB(
            model=args.model,
            num_instances=getattr(args, "num_instances", 1),
            pic_memory_utilization=getattr(args, "pic_memory_utilization", 0.3),
            pbc_memory_utilization=getattr(args, "pbc_memory_utilization", 0.3),
            pic_separated=getattr(args, "pic_separated", False),
            disable_log_stats=getattr(args, "disable_log_stats", False),
        )
    
    app_instance.state.comb_engine = async_comb_engine
    return app_instance


async def run_server(
    args: Namespace,
    comb_engine: Optional[AsyncCOMB] = None,
    **uvicorn_kwargs: Any
) -> None:
    """Run the async COMB API server.
    
    Args:
        args: Command-line arguments.
        comb_engine: Optional pre-initialized AsyncCOMB engine.
        **uvicorn_kwargs: Additional keyword arguments for uvicorn.
    """
    logger.info("COMB API server starting...")
    logger.info("Arguments: %s", args)
    
    app_instance = await init_app(args, comb_engine)
    assert async_comb_engine is not None
    
    # For now, we use a simple approach with uvicorn.
    # In production, you might want to use a more sophisticated server setup.
    import uvicorn
    
    config = uvicorn.Config(
        app=app_instance,
        host=getattr(args, "host", "127.0.0.1"),
        port=getattr(args, "port", 8000),
        log_level=getattr(args, "log_level", "info"),
        **uvicorn_kwargs,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="COMB Async HTTP API Server"
    )
    
    # Server configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the COMB model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default="",
        help="Root path for the API (useful when behind a proxy)"
    )
    
    # COMB configuration
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1,
        help="Number of inference engine instances"
    )
    parser.add_argument(
        "--pic-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization for PIC (0-1)"
    )
    parser.add_argument(
        "--pbc-memory-utilization",
        type=float,
        default=0.3,
        help="GPU memory utilization for prefix-based cache (0-1)"
    )
    parser.add_argument(
        "--pic-separated",
        action="store_true",
        help="Whether to assign different GPUs for PIC and LLM engine"
    )
    parser.add_argument(
        "--disable-log-stats",
        action="store_true",
        help="Disable logging statistics"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Model context length (prompt and output)"
    )
    
    args = parser.parse_args()
    
    # Run the server
    asyncio.run(run_server(args))
