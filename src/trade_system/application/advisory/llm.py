from __future__ import annotations

import json
import os
import logging
from dataclasses import asdict
from typing import Any

import requests
from trade_system.config import Settings
from trade_system.application.advisory.rulebook import OPTION_BUYER_RULEBOOK

LOGGER = logging.getLogger(__name__)

class LlmAdvisorClient:
    """
    The 'Brain' of the AI Swarm. 
    Configurable to use high-capacity models like Gemini 1.5 Pro or Gemma.
    """
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        settings = Settings.load()
        self.provider = provider or settings.llm.provider
        self.api_key = api_key or settings.llm.api_key
        self.model = model or settings.llm.model
        self.base_url = settings.llm.base_url
        self.timeout_seconds = timeout_seconds

    def configured(self) -> bool:
        return bool(self.api_key)

    async def complete(self, prompt: str) -> str:
        """General purpose completion for any brain task."""
        if not self.configured():
            LOGGER.warning("LLM not configured. Skipping brain task.")
            return "Brain not configured."

        if self.provider == "gemini":
            return await self._call_gemini(prompt)
        else:
            return self._call_openai(prompt)

    def suggest(self, *, context: Any, proposal: Any, advice: Any) -> str:
        """Specific trade suggestion logic."""
        if not self.configured():
            return "No LLM analysis available (Not configured)."

        prompt = self._build_prompt(context=context, proposal=proposal, advice=advice)
        
        # Use sync wrapper for legacy suggest calls
        import asyncio
        try:
            return asyncio.run(self.complete(prompt))
        except RuntimeError: # Already in loop
            return "LLM fetch failed in sync context."

    def _build_prompt(self, *, context: Any, proposal: Any, advice: Any) -> str:
        return (
            "You are a risk-first option buying assistant.\n\n"
            "Follow this rulebook strictly.\n"
            f"{OPTION_BUYER_RULEBOOK}\n\n"
            "Return a concise trader note with:\n"
            "1. verdict\n"
            "2. setup summary\n"
            "3. key risks\n"
            "4. revised trade idea if needed\n\n"
            f"Context:\n{json.dumps(asdict(context), indent=2, default=str)}\n\n"
            f"Proposal:\n{json.dumps(asdict(proposal), indent=2, default=str)}\n\n"
            f"Deterministic audit:\n{json.dumps(asdict(advice), indent=2, default=str)}\n"
        )

    def _call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds
            )
            
            completion = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=0.95,
                max_tokens=8192,
                stream=True
            )
            
            result = ""
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content
                    
            if not result:
                return "Error: Brain returned an empty response."
            return result
        except Exception as e:
            LOGGER.error(f"OpenAI Brain error: {e}")
            return f"Error connecting to OpenAI brain: {e}"

    async def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini 1.5 Pro / Gemma brain."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        try:
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            LOGGER.error(f"Gemini Brain error: {e}")
            return f"Error connecting to Gemini brain: {e}"
