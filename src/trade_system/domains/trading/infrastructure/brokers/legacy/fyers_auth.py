import json
import requests
try:
    import pyotp
except ImportError:
    pyotp = None
import hashlib
import logging
import os
from datetime import datetime, date
from urllib import parse
from pathlib import Path
from trade_system.shared.config import Settings

logger = logging.getLogger(__name__)

class FyersAuthenticator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url_vagator = "https://api-t2.fyers.in/vagator/v2"
        self.base_url_api = "https://api-t1.fyers.in/api/v3"

    def _sha256_hash(self, client_id_prefix: str, app_type: str, secret_key: str) -> str:
        message = f"{client_id_prefix}-{app_type}:{secret_key}"
        return hashlib.sha256(message.encode()).hexdigest()

    def generate_access_token(self) -> str:
        """
        Generates a new Fyers access token using TOTP and Pin.
        Returns the full access token string.
        """
        logger.info("Starting automated Fyers token generation...")
        
        try:
            # 1. Send Login OTP (Internal step for Fyers)
            payload_step1 = {"fy_id": self.settings.fyers.user_id, "app_id": "2"}
            res1 = requests.post(f"{self.base_url_vagator}/send_login_otp", json=payload_step1)
            res1.raise_for_status()
            request_key = res1.json()["request_key"]

            # 2. Verify TOTP
            if pyotp is None:
                raise RuntimeError("pyotp is required for automated token refresh.")
            
            totp_gen = pyotp.TOTP(self.settings.fyers.totp_secret)
            totp = totp_gen.now()
            
            payload_step2 = {"request_key": request_key, "otp": totp}
            res2 = requests.post(f"{self.base_url_vagator}/verify_otp", json=payload_step2)
            if res2.status_code != 200:
                logger.error(f"TOTP verification failed (HTTP {res2.status_code}): {res2.text}")
                logger.error(f"Used User ID: {self.settings.fyers.user_id}")
                res2.raise_for_status()
            
            request_key = res2.json()["request_key"]

            # 3. Verify PIN
            payload_step3 = {
                "request_key": request_key,
                "identity_type": "pin",
                "identifier": self.settings.fyers.pin
            }
            res3 = requests.post(f"{self.base_url_vagator}/verify_pin", json=payload_step3)
            res3.raise_for_status()
            internal_access_token = res3.json()["data"]["access_token"]

            # 4. Get Auth Code
            # Extract APP_ID prefix (e.g., ALHT0QX10K from ALHT0QX10K-100)
            client_id_parts = self.settings.fyers.client_id.split('-')
            app_id_prefix = client_id_parts[0]
            app_type = client_id_parts[1] if len(client_id_parts) > 1 else "100"

            payload_step4 = {
                "fyers_id": self.settings.fyers.user_id,
                "app_id": app_id_prefix,
                "redirect_uri": self.settings.fyers.redirect_uri,
                "appType": app_type,
                "code_challenge": "",
                "state": "sample_state",
                "scope": "",
                "nonce": "",
                "response_type": "code",
                "create_cookie": True
            }
            headers = {'Authorization': f'Bearer {internal_access_token}'}
            res4 = requests.post(f"{self.base_url_api}/token", json=payload_step4, headers=headers)
            
            # Fyers returns 308 for this specific call to redirect
            if res4.status_code != 308:
                logger.error(f"Auth code step failed: {res4.text}")
                res4.raise_for_status()

            url = res4.json()["Url"]
            auth_code = parse.parse_qs(parse.urlparse(url).query)['auth_code'][0]

            # 5. Validate Auth Code to get final API Access Token
            app_id_hash = self._sha256_hash(app_id_prefix, app_type, self.settings.fyers.secret_key)
            payload_step5 = {
                "grant_type": "authorization_code",
                "appIdHash": app_id_hash,
                "code": auth_code,
            }
            res5 = requests.post(f"{self.base_url_api}/validate-authcode", json=payload_step5)
            res5.raise_for_status()
            
            final_token = res5.json()["access_token"]
            
            # Fyers V3 access token format is usually "CLIENT_ID:TOKEN"
            # But the SDK handles just the token part if initialized correctly
            # Based on totp_auth.py, it constructs it as APP_ID-APP_TYPE:TOKEN
            full_access_token = f"{app_id_prefix}-{app_type}:{final_token}"
            
            logger.info("Successfully generated new Fyers access token.")
            return full_access_token

        except Exception as e:
            logger.error(f"Failed to generate Fyers token: {e}")
            raise

    def update_env_file(self, new_token: str, env_path: str = ".env"):
        """Updates the FYERS_ACCESS_TOKEN in the .env file."""
        import re
        
        # Try to find .env in current or parent directories
        target_path = Path(env_path)
        if not target_path.exists():
            # Try project root
            root_env = Path(__file__).resolve().parents[4] / ".env"
            if root_env.exists():
                target_path = root_env
            else:
                logger.warning(f"{env_path} not found. Skipping file update.")
                return

        # We store the full token (appid:token) in .env for consistency with Fyers V3
        env_token = new_token

        content = target_path.read_text()
        pattern = r'^FYERS_ACCESS_TOKEN\s*=\s*.*$'
        replacement = f'FYERS_ACCESS_TOKEN="{env_token}"'
        
        new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)
        
        if count == 0:
            # Append if not found
            if not new_content.endswith('\n'):
                new_content += '\n'
            new_content += f'{replacement}\n'
        
        target_path.write_text(new_content)
        logger.info(f"Updated {target_path} with new access token.")


class FyersAuthService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.authenticator = FyersAuthenticator(settings)

    def _is_token_from_today(self, token: str) -> bool:
        if not token:
            return False
        try:
            token_part = token.split(":")[-1]
            parts = token_part.split(".")
            if len(parts) != 3:
                return False
            payload_b64 = parts[1]
            padding = "=" * (4 - len(payload_b64) % 4)
            import base64
            import json
            payload_json = base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8")
            payload = json.loads(payload_json)
            iat = payload.get("iat")
            if iat:
                dt = datetime.fromtimestamp(iat)
                return dt.date() == date.today()
        except Exception as e:
            logger.warning(f"Failed to parse token JWT: {e}")
        return False

    def get_valid_token(self, force_refresh: bool = False) -> str | None:
        """
        Retrieves a valid token. 
        Checks cache first, then environment, and finally triggers TOTP if needed.
        """
        # 1. Try Cache File first (always prefer the shared token on disk if valid)
        token = None
        if not force_refresh:
            token = self.read_cached_token()
            if token and not self._is_token_from_today(token):
                token = None
            
        # 2. Try Memory/Config (loaded from env)
        if not token:
            env_token = self.settings.fyers.access_token
            if env_token and self._is_token_from_today(env_token):
                token = env_token
            
        # 3. Trigger TOTP only if absolutely required
        if not token or force_refresh:
            logger.info("Shared Auth: No valid token found or token expired. Triggering once-per-day TOTP flow.")
            try:
                token = self.authenticate_totp()
                # Update .env so other processes pick it up
                self.authenticator.update_env_file(token)
            except Exception as e:
                logger.error(f"Shared Auth: TOTP generation failed: {e}")
                return None
                
        return token

    def read_cached_token(self) -> str | None:
        path = self.settings.fyers.token_path
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                token = payload.get("access_token")
                
                # Basic sanity check: Is the token from today?
                # Fyers tokens expire daily at midnight.
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                if mtime.date() != date.today():
                    logger.info("Shared Auth: Cached token is from a previous day. Expiring.")
                    return None
                    
                return token
            except Exception:
                logger.warning("Failed to read cached FYERS token file.")
        return None

    def authenticate_totp(self) -> str:
        token = self.authenticator.generate_access_token()
        self._cache_token(token, source="totp")
        return token

    def authenticate_browser(self, auth_code: str | None = None, *, open_browser: bool = True) -> str:
        return self.authenticate_totp()

    def _cache_token(self, token: str, source: str = "totp") -> None:
        object.__setattr__(self.settings.fyers, "access_token", token)
        self.settings.fyers.token_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"access_token": token, "source": source}
        self.settings.fyers.token_path.write_text(json.dumps(payload, indent=2))
