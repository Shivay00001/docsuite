from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class LicensePlan(BaseModel):
    name: str  # free, pro, enterprise
    max_pages_per_month: int
    features: List[str]
    api_rate_limit: int = 60  # requests per minute
