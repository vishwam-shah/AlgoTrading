"""
Loguru configuration for V3 pipeline.

- File-only logging (no console output except errors)
- Rotating handler (50 MB max per file)
- 30-day retention with gzip compression
- Thread-safe enqueuing for parallel workers
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(run_id: str, log_dir: Path, level: str = "INFO") -> None:
    """
    Configure loguru for a pipeline run.

    Removes default stderr sink, adds rotating file sink with errors-only stderr.

    Args:
        run_id: Pipeline run ID (e.g., "20260408_140735")
        log_dir: Directory for logs (e.g., V3/06_results/runs/{run_id}/)
        level: Log level ("DEBUG" for verbose, "INFO" for production)
    """
    # Remove default stderr handler (stops all console output)
    logger.remove()

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"run_{run_id}.log"

    # File sink: all messages ≥ level, rotating, compressed
    logger.add(
        str(log_file),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} | {message}",
        rotation="50 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,       # Thread-safe for parallel workers
        catch=True,         # Catch exceptions in logging itself
    )

    # Errors-only to stderr (so crashes are visible to user)
    logger.add(
        lambda msg: sys.stderr.write(msg),
        level="ERROR",
        format="{time:HH:mm:ss} | ❌ ERROR | {message}",
        colorize=False,
    )

    logger.info(f"Logging initialized | run_id={run_id} | log_file={log_file}")
