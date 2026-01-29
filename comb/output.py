# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the COMB project
from pydantic import BaseModel

class RequestOutput(BaseModel):
    token_ids: list[int]
    finish_reason: str | None

    # Stats
    num_prompt_tokens: int = 0
    num_generation_tokens: int = 0
    first_token_ts: float = 0.0
    last_token_ts: float = 0.0
    first_token_latency: float = 0.0

    def finished(self) -> bool:
        return self.finish_reason is not None

    def time_per_output_token(self) -> float:
        if n := self.num_generation_tokens:
            return (self.last_token_ts - self.first_token_ts) / n
        
        return 0