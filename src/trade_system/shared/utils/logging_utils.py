from __future__ import annotations

import logging
import json
import datetime
import traceback
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        if record.exc_info:
            log_data["exception"] = "".join(traceback.format_exception(*record.exc_info))
            
        return json.dumps(log_data)


from pathlib import Path
from logging.handlers import RotatingFileHandler

def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    root_dir = Path(__file__).resolve().parents[4]
    log_dir = root_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "trade_system.log"

    stream_handler = logging.StreamHandler()
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)

    if as_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[stream_handler, file_handler],
        force=True
    )
