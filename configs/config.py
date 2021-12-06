from dataclasses import dataclass
import torch


@dataclass
class TaskConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size: int = 64
    dict_size: int = 51
    emb_dim: int = 384
    n_blocks1: int = 6
    n_blocks2: int = 6
    n_heads: int = 2
    conv_size: int = 1536
    kernel_size: int = 3
    duration_size: int = 3
    attn_size: int = 2
    output_size: int = 80
    dropout: float = 0.1
    alpha: float = 1.0
    n_epochs: int = 80
    lr: float = 1e-4
    project_name: str = 'tts'
    show_every: int = 5
    warmup: int = 4000
