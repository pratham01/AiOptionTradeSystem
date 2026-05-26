"""
Notifier Port Interface (Layer 1).

Defines the contract for sending notifications (e.g., via Telegram, Email, Slack, WhatsApp),
decoupling the execution loop from specific notification delivery integrations.
"""

from __future__ import annotations
from typing import Protocol

class Notifier(Protocol):
    """Protocol for system notifications."""
    
    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a text-based message.

        Args:
            text: Message body content.
            parse_mode: Parsing mode (e.g., 'HTML', 'Markdown').

        Returns:
            True if sent successfully, False otherwise.
        """
        ...
        
    def send(self, text: str, parse_mode: str = "HTML") -> bool:
        """Alias for send_message."""
        ...
        
    def send_photo(self, photo_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        Send an image/photo along with a caption.

        Args:
            photo_path: Absolute local path to the image file.
            caption: Optional image description text.
            parse_mode: Parsing mode for caption formatting.

        Returns:
            True if sent successfully, False otherwise.
        """
        ...
