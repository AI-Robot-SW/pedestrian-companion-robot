# Optional: these modules live under examples/ in this repo; allow package to load without them
try:
    from .context_provider import ContextProvider
except ImportError:
    ContextProvider = None  # type: ignore[misc, assignment]

try:
    from .io_provider import IOProvider
except ImportError:
    IOProvider = None  # type: ignore[misc, assignment]

try:
    from .teleops_status_provider import (
        BatteryStatus,
        CommandStatus,
        TeleopsStatus,
        TeleopsStatusProvider,
    )
except ImportError:
    BatteryStatus = None  # type: ignore[misc, assignment]
    CommandStatus = None  # type: ignore[misc, assignment]
    TeleopsStatus = None  # type: ignore[misc, assignment]
    TeleopsStatusProvider = None  # type: ignore[misc, assignment]

__all__ = [
    "ContextProvider",
    "IOProvider",
    "TeleopsStatusProvider",
    "CommandStatus",
    "BatteryStatus",
    "TeleopsStatus",
]