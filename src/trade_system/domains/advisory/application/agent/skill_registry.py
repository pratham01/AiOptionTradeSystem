"""
SkillRegistry — Utility to load and manage dynamic trading skills created by the SkillCreatorAgent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

LOGGER = logging.getLogger(__name__)

class SkillRegistry:
    """
    Loads and provides Markdown-based skills to AI Agents.
    """

    def __init__(self, skills_dir: str = "data/skills") -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def get_all_skills_text(self) -> str:
        """
        Concatenates all skills into a single string for LLM prompt injection.
        """
        skill_files = list(self.skills_dir.glob("*.md"))
        if not skill_files:
            return "No specialized skills discovered yet."

        content = []
        for file in skill_files:
            try:
                text = file.read_text().strip()
                content.append(f"--- SKILL: {file.stem} ---\n{text}")
            except Exception as e:
                LOGGER.error(f"Failed to read skill {file}: {e}")

        return "\n\n".join(content)

    def list_skills(self) -> List[str]:
        """Returns list of skill names."""
        return [f.stem for f in self.skills_dir.glob("*.md")]
