"""Optional neural components (require the ``neural`` extra: torch)."""

try:
    import torch  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "roomkit.neural requires torch; install it with `pip install roomkit[neural]`"
    ) from exc

from roomkit.neural.config import EncoderConfig
from roomkit.neural.encoder import WhisperStyleEncoder
from roomkit.neural.pooling import statistics_pool

__all__ = ["EncoderConfig", "WhisperStyleEncoder", "statistics_pool"]
