#!/usr/bin/env python3
"""Encrypt/decrypt the .env secrets file for production-local deployment.

Usage:
    # Encrypt .env → .secrets.enc
    python scripts/encrypt_secrets.py encrypt --master-key "your-strong-password"

    # Decrypt .secrets.enc → stdout (for debugging)
    python scripts/encrypt_secrets.py decrypt --master-key "your-strong-password"

    # Encrypt with master key from environment variable
    MASTER_KEY="your-strong-password" python scripts/encrypt_secrets.py encrypt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt/decrypt secrets files")
    sub = parser.add_subparsers(dest="command", required=True)

    # Encrypt command
    enc = sub.add_parser("encrypt", help="Encrypt .env → .secrets.enc")
    enc.add_argument("--env-file", type=str, default=".env", help="Source .env file")
    enc.add_argument("--output", type=str, default=".secrets.enc", help="Output encrypted file")
    enc.add_argument("--master-key", type=str, default=None, help="Master password (or set MASTER_KEY env var)")

    # Decrypt command
    dec = sub.add_parser("decrypt", help="Decrypt .secrets.enc → stdout")
    dec.add_argument("--input", type=str, default=".secrets.enc", help="Encrypted file to decrypt")
    dec.add_argument("--master-key", type=str, default=None, help="Master password (or set MASTER_KEY env var)")

    args = parser.parse_args()

    import os
    master_key = args.master_key or os.getenv("MASTER_KEY")
    if not master_key:
        print("ERROR: Master key required. Use --master-key or set MASTER_KEY env var.", file=sys.stderr)
        sys.exit(1)

    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from trade_system.shared.config.secrets_provider import EncryptedFileSecretsProvider

    if args.command == "encrypt":
        if not Path(args.env_file).exists():
            print(f"ERROR: {args.env_file} not found.", file=sys.stderr)
            sys.exit(1)
        EncryptedFileSecretsProvider.encrypt_env_file(args.env_file, args.output, master_key)
        print(f"✅ Encrypted {args.env_file} → {args.output}")

    elif args.command == "decrypt":
        if not Path(args.input).exists():
            print(f"ERROR: {args.input} not found.", file=sys.stderr)
            sys.exit(1)
        provider = EncryptedFileSecretsProvider(args.input, master_key)
        for key in sorted(provider.list_keys()):
            print(f"{key}={provider.get_secret(key)}")


if __name__ == "__main__":
    main()
