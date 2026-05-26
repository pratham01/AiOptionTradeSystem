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


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler()
    if as_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True
    )
