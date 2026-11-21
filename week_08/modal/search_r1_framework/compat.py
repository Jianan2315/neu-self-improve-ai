"""Expose an adapter-loading field already read by veRL v0.6.1 CriticWorker."""
from dataclasses import dataclass
from typing import Optional
from verl.workers.config.critic import FSDPCriticModelCfg


@dataclass
class CriticModelConfig(FSDPCriticModelCfg):
    # The worker supports this field, but its v0.6.1 config dataclass omits it.
    lora_adapter_path: Optional[str] = None
