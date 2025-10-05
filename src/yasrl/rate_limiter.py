from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from .auth import get_website_id

def get_rate_limit_key(request: Request):
    """
    Determines the key for rate limiting.
    It uses the website ID from the API key.
    """
    try:
        # This is a bit of a hack, as we can't directly call Depends here.
        # We manually extract the api key and resolve it to a website_id.
        api_key = request.headers.get("x-api-key")
        if api_key:
            from .auth import VALID_API_KEYS
            website_id = VALID_API_KEYS.get(api_key)
            if website_id:
                return website_id
    except Exception:
        pass  # Fallback to IP address if anything goes wrong

    return get_remote_address(request)


limiter = Limiter(key_func=get_rate_limit_key)