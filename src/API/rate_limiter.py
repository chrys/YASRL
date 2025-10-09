from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import jwt
import os

# Configuration should be consistent with auth.py
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"

def get_rate_limit_key(request: Request) -> str:
    """
    Determines the key for rate limiting.
    It tries to use the 'website_id' from the JWT token.
    If the token is not present or invalid, it falls back to the client's IP address.
    """
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            website_id = payload.get("website_id")
            if website_id:
                print(f"DEBUG: Rate limiting by website_id: {website_id}")
                return website_id
    except (jwt.PyJWTError, IndexError):
        # Fallback if token is invalid or header is malformed
        pass
    
    remote_address = get_remote_address(request)
    print(f"DEBUG: Rate limiting by IP address: {remote_address}")
    return remote_address

limiter = Limiter(key_func=get_rate_limit_key)
