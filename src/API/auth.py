import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

class TokenData(BaseModel):
    user_id: str
    website_id: str

class TokenResponse(BaseModel):
    access_token: str

class TokenRequest(BaseModel):
    username: str
    password: str


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Simplified auth - only use JWT token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    print(f"DEBUG: Received JWT token: {token[:50]}...")
    
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"DEBUG: JWT payload: {payload}")
        user_id: str = payload.get("sub")
        website_id: str = payload.get("website_id")
        project_id: str = payload.get("project_id")  
        project_name: str = payload.get("project_name")  
        
        if user_id is None or website_id is None:
            print("DEBUG: Missing user_id or website_id in JWT payload")
            raise credentials_exception
            
        print(f"DEBUG: Extracted user_id={user_id}, website_id={website_id}, project_id={project_id}, project_name={project_name}")
        
        # Return all the fields from JWT payload
        return {
            "user_id": user_id, 
            "website_id": website_id,
            "project_id": project_id,
            "project_name": project_name
        }
        
    except (jwt.PyJWTError, ValidationError) as e:
        print(f"DEBUG: JWT validation error: {e}")
        raise credentials_exception

