"""
Data validation schemas — Pydantic models that guard every external data entry point.

RULE: No raw broker API response or database row should ever flow into an indicator
or agent without first being validated through one of these schemas.

This prevents:
- Silent NoneType propagation (the #1 cause of corrupted signals)
- Type mismatches (str price instead of float)
- Missing required fields causing KeyError crashes mid-calculation
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

if PYDANTIC_AVAILABLE:
    class TickSchema(BaseModel):
        """Validates a single websocket tick before processing."""
        symbol: str
        ltp: float = Field(gt=0, description="Last traded price must be positive")
        timestamp: datetime
        volume: float = Field(ge=0, default=0.0)

        @field_validator("ltp", "volume", mode="before")
        @classmethod
        def coerce_to_float(cls, v):
            if v is None:
                return 0.0
            return float(v)


    class CandleSchema(BaseModel):
        """Validates OHLCV candle data from broker history APIs."""
        timestamp: datetime
        open: float = Field(gt=0)
        high: float = Field(gt=0)
        low: float = Field(gt=0)
        close: float = Field(gt=0)
        volume: float = Field(ge=0, default=0.0)

        @field_validator("open", "high", "low", "close", "volume", mode="before")
        @classmethod
        def coerce_to_float(cls, v):
            if v is None:
                raise ValueError("OHLCV values cannot be None")
            return float(v)

        @model_validator(mode="after")
        def validate_price_integrity(self) -> "CandleSchema":
            """Enforce OHLC relationships to catch bad broker data."""
            if self.high < self.low:
                raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
            if not (self.low <= self.open <= self.high):
                raise ValueError(f"Open ({self.open}) must be between Low ({self.low}) and High ({self.high})")
            if not (self.low <= self.close <= self.high):
                raise ValueError(f"Close ({self.close}) must be between Low ({self.low}) and High ({self.high})")
            return self


    class OptionStrikeSchema(BaseModel):
        """Validates a single option chain row."""
        timestamp: datetime
        spot_price: float = Field(ge=0)
        strike: float = Field(gt=0)
        option_type: str  # "CE" or "PE"
        symbol: str
        ltp: float = Field(ge=0, default=0.0)
        volume: float = Field(ge=0, default=0.0)
        open_interest: float = Field(ge=0, default=0.0)
        iv: Optional[float] = None
        delta: Optional[float] = None
        gamma: Optional[float] = None
        theta: Optional[float] = None
        vega: Optional[float] = None

        @field_validator("option_type")
        @classmethod
        def validate_option_type(cls, v: str) -> str:
            if v.upper() not in ("CE", "PE"):
                raise ValueError(f"option_type must be CE or PE, got: {v}")
            return v.upper()

        @field_validator("ltp", "volume", "open_interest", mode="before")
        @classmethod
        def coerce_non_negative(cls, v):
            if v is None:
                return 0.0
            val = float(v)
            return max(val, 0.0)

        @field_validator("iv", "delta", "gamma", "theta", "vega", mode="before")
        @classmethod
        def coerce_optional_float(cls, v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None


    def validate_candles(raw_candles: list[dict]) -> tuple[list[CandleSchema], list[str]]:
        """
        Validate a list of raw candle dicts.
        Returns (valid_candles, list_of_error_messages).
        Errors are logged but do not crash the pipeline.
        """
        valid = []
        errors = []
        for i, c in enumerate(raw_candles):
            try:
                valid.append(CandleSchema(**c))
            except Exception as e:
                errors.append(f"Candle[{i}] invalid: {e}")
        return valid, errors


    def validate_option_chain(raw_rows: list[dict]) -> tuple[list[OptionStrikeSchema], list[str]]:
        """
        Validate a list of raw option chain rows.
        Returns (valid_rows, list_of_error_messages).
        """
        valid = []
        errors = []
        for i, row in enumerate(raw_rows):
            try:
                valid.append(OptionStrikeSchema(**row))
            except Exception as e:
                errors.append(f"OptionRow[{i}] invalid: {e}")
        return valid, errors

else:
    # Graceful fallback when pydantic is not installed
    # This avoids a hard crash while still being explicit about the gap
    import warnings
    warnings.warn(
        "pydantic is not installed. Data validation is DISABLED. "
        "Install it with: pip install pydantic>=2.0",
        RuntimeWarning, stacklevel=2
    )

    class TickSchema:  # type: ignore[no-redef]
        def __init__(self, **data): self.__dict__.update(data)

    class CandleSchema:  # type: ignore[no-redef]
        def __init__(self, **data): self.__dict__.update(data)

    class OptionStrikeSchema:  # type: ignore[no-redef]
        def __init__(self, **data): self.__dict__.update(data)

    def validate_candles(raw_candles):  # type: ignore[misc]
        return raw_candles, []

    def validate_option_chain(raw_rows):  # type: ignore[misc]
        return raw_rows, []
