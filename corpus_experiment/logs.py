import logging
import time


class DeltaFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            "%(levelname).1s %(asctime)s +%(delta).2fs %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._last: float | None = None

    def format(self, record: logging.LogRecord) -> str:
        now = time.monotonic()
        record.delta = 0.0 if self._last is None else now - self._last
        self._last = now
        return super().format(record)


def configure_logging() -> None:
    """Configure one stderr handler shared by the script and library loggers."""
    handler = logging.StreamHandler()
    handler.setFormatter(DeltaFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
