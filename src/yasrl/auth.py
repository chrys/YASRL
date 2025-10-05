import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, ValidationError

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# This would typically be stored in a secure database
VALID_API_KEYS = {
    "website1_key": "website1",
    "website2_key": "website2",
}

api_key_header = APIKeyHeader(name="X-API-Key")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class TokenData(BaseModel):
    user_id: str
    website_id: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_website_id(api_key: str = Depends(api_key_header)):
    if api_key in VALID_API_KEYS:
        return VALID_API_KEYS[api_key]
    else:
        raise HTTPException(status_code=401, detail="Invalid API Key")


async def get_current_user(
    token: str = Depends(oauth2_scheme), website_id: str = Depends(get_website_id)
):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        token_data = TokenData(user_id=user_id, website_id=website_id)
    except (jwt.PyJWTError, ValidationError):
        raise credentials_exception

    # In a real application, you would fetch the user from the database
    # For this example, we'll just return the token data
    return {"user_id": token_data.user_id, "website_id": token_data.website_id}