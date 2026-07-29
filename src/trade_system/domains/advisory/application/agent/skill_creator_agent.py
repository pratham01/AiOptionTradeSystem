"""
SkillCreatorAgent — Meta-agent that writes new rules and skills for the system.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from trade_system.domains.advisory.application.advisory.llm import LlmAdvisorClient

LOGGER = logging.getLogger(__name__)

class SkillCreatorAgent:
    """
    Takes insights from post-market analysis (what worked, what failed) and 
    translates them into formal, reusable 'skills' or rules for the AI agents to follow.
    """

    def __init__(self, llm_client: LlmAdvisorClient | None = None) -> None:
        self.llm = llm_client or LlmAdvisorClient()
        self.skills_dir = Path("data/skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    async def create_skill(self, insight: str, context: str) -> str:
        """
        Convert a post-market insight into a formalized skill document.
        """
        LOGGER.info("SkillCreatorAgent: Analyzing insight to create a new skill...")
        
        if not self.llm.configured():
            LOGGER.warning("LLM not configured. Cannot generate complex skill.")
            return "Failed: LLM not configured."

        prompt = (
            "You are a Meta-Agent responsible for programming other trading AI agents.\n"
            "Based on the following post-market insight, create a formalized trading 'Skill'.\n\n"
            f"Insight: {insight}\n"
            f"Context: {context}\n\n"
            "Format the output as a Markdown document with:\n"
            "1. # Skill Name (e.g. 'high-vix-mean-reversion')\n"
            "2. ## Trigger Conditions (when to use this skill)\n"
            "3. ## Execution Rules (exact rules for the agent)\n"
            "4. ## Risk Management (stop loss, sizing)\n\n"
            "Output ONLY the markdown content."
        )

        try:
            skill_markdown = await self.llm.complete(prompt)
            
            # Extract a filename from the first line (e.g., # Skill Name)
            first_line = skill_markdown.split('\n')[0]
            file_name = first_line.replace('#', '').strip().lower().replace(' ', '-') + ".md"
            if not file_name.endswith('.md') or file_name == ".md":
                file_name = "new_skill.md"

            file_path = self.skills_dir / file_name
            
            # Save the skill
            with open(file_path, "w") as f:
                f.write(skill_markdown)
                
            LOGGER.info(f"Successfully created new skill: {file_path}")
            return f"Skill created at {file_path}"

        except Exception as e:
            LOGGER.error(f"Failed to create skill: {e}")
            return f"Error: {e}"
