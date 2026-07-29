from __future__ import annotations

import json
import os
from typing import Any

import requests


class ResearchLlmClient:
    def __init__(
        self,
        *,
        free_base_url: str | None = None,
        free_model: str | None = None,
        free_api_key: str | None = None,
        paid_base_url: str | None = None,
        paid_model: str | None = None,
        paid_api_key: str | None = None,
        timeout_seconds: int = 45,
    ) -> None:
        self.free_base_url = free_base_url or os.getenv("LLM_FREE_BASE_URL", "").strip()
        self.free_model = free_model or os.getenv("LLM_FREE_MODEL", "").strip()
        self.free_api_key = free_api_key or os.getenv("LLM_FREE_API_KEY", "").strip()
        self.paid_base_url = paid_base_url or os.getenv("LLM_PAID_BASE_URL", "").strip() or os.getenv("LLM_BASE_URL", "").strip() or "https://api.openai.com/v1/responses"
        self.paid_model = paid_model or os.getenv("LLM_PAID_MODEL", "").strip() or os.getenv("LLM_MODEL", "").strip() or "gpt-4.1-mini"
        self.paid_api_key = paid_api_key or os.getenv("LLM_PAID_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
        self.timeout_seconds = timeout_seconds

    def provider_status(self) -> dict[str, bool]:
        return {
            "free_configured": bool(self.free_base_url and self.free_model),
            "paid_configured": bool(self.paid_api_key and self.paid_model and self.paid_base_url),
        }

    def review(self, prompt: str) -> tuple[str, str]:
        status = self.provider_status()
        try:
            if status["free_configured"]:
                return "free", self._call(
                    base_url=self.free_base_url,
                    model=self.free_model,
                    api_key=self.free_api_key or None,
                    prompt=prompt,
                )
            if status["paid_configured"]:
                return "paid", self._call(
                    base_url=self.paid_base_url,
                    model=self.paid_model,
                    api_key=self.paid_api_key,
                    prompt=prompt,
                )
        except Exception as e:
            return "error", f"LLM Call failed: {str(e)}"
        return "deterministic", "No LLM configured. Deterministic research review only."

    def _call(self, *, base_url: str, model: str, api_key: str | None, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = requests.post(
            base_url,
            headers=headers,
            json={"model": model, "input": prompt},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return self._extract_text(response.json())

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()
        output = payload.get("output") or []
        texts: list[str] = []
        for item in output:
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    texts.append(text)
        if texts:
            return "\n".join(texts).strip()
        return json.dumps(payload, indent=2, default=str)
