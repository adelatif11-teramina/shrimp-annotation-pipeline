"""
JWT Authentication and Authorization
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.settings import get_settings

settings = get_settings()

class TokenData(BaseModel):
    """JWT token data structure"""
    user_id: int
    username: str
    email: Optional[str] = None
    role: str
    permissions: List[str] = []
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for revocation

class UserCreate(BaseModel):
    """User creation model"""
    username: str
    email: str
    password: str
    role: str = "annotator"

class UserResponse(BaseModel):
    """User response model (no password)"""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse

# Role-based permissions
ROLE_PERMISSIONS = {
    "admin": [
        "read:all",
        "write:all", 
        "delete:all",
        "manage:users",
        "manage:system",
        "export:data",
        "view:analytics"
    ],
    "reviewer": [
        "read:annotations",
        "read:candidates",
        "write:annotations",
        "review:annotations",
        "export:data",
        "view:analytics"
    ],
    "annotator": [
        "read:annotations",
        "read:candidates", 
        "write:annotations",
        "read:own_stats"
    ],
    "readonly": [
        "read:annotations",
        "read:candidates",
        "read:own_stats"
    ]
}

class PasswordManager:
    """Secure password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt"""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class JWTManager:
    """JWT token creation and validation"""
    
    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.jwt_refresh_token_expire_days
        
        # Token revocation storage (in production, use Redis)
        self.revoked_tokens: set = set()
    
    def create_access_token(
        self, 
        user_id: int, 
        username: str, 
        email: str, 
        role: str,
        extra_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create JWT access token"""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "role": role,
            "permissions": ROLE_PERMISSIONS.get(role, []),
            "exp": expire,
            "iat": now,
            "type": "access"
        }
        
        if extra_claims:
            payload.update(extra_claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(
        self, 
        user_id: int, 
        username: str
    ) -> str:
        """Create JWT refresh token"""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "user_id": user_id,
            "username": username,
            "exp": expire,
            "iat": now,
            "type": "refresh"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                raise HTTPException(
                    status_code=401,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Validate token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token type"
                )
            
            # Extract token data
            token_data = TokenData(
                user_id=payload["user_id"],
                username=payload["username"],
                email=payload.get("email"),
                role=payload["role"],
                permissions=payload.get("permissions", []),
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
                jti=payload.get("jti")
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
    
    def verify_refresh_token(self, token: str) -> Dict[str, Any]:
        """Verify refresh token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid refresh token"
                )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Refresh token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=401,
                detail="Invalid refresh token"
            )
    
    def revoke_token(self, token: str):
        """Revoke a token (add to blacklist)"""
        self.revoked_tokens.add(token)
    
    def refresh_access_token(self, refresh_token: str, user_data: Dict[str, Any]) -> str:
        """Create new access token from refresh token"""
        # Verify refresh token
        payload = self.verify_refresh_token(refresh_token)
        
        # Create new access token
        return self.create_access_token(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            role=user_data["role"]
        )

# Global JWT manager instance
jwt_manager = JWTManager()
password_manager = PasswordManager()

# FastAPI security
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> TokenData:
    """Get current authenticated user from JWT token"""
    token = credentials.credentials
    return jwt_manager.verify_token(token)

async def get_current_active_user(
    current_user: TokenData = Depends(get_current_user)
) -> TokenData:
    """Get current active user (could add active status check here)"""
    return current_user

def require_permissions(*required_permissions: str):
    """Decorator to require specific permissions"""
    def permission_checker(current_user: TokenData = Depends(get_current_active_user)):
        user_permissions = set(current_user.permissions)
        required_perms = set(required_permissions)
        
        # Check if user has all required permissions
        if not required_perms.issubset(user_permissions):
            missing_perms = required_perms - user_permissions
            raise HTTPException(
                status_code=403,
                detail=f"Missing required permissions: {', '.join(missing_perms)}"
            )
        
        return current_user
    
    return permission_checker

def require_role(*allowed_roles: str):
    """Decorator to require specific roles"""
    def role_checker(current_user: TokenData = Depends(get_current_active_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        
        return current_user
    
    return role_checker

# Common permission dependencies
require_admin = require_role("admin")
require_reviewer_or_admin = require_role("reviewer", "admin")
require_annotator_access = require_permissions("read:annotations", "write:annotations")
require_export_permission = require_permissions("export:data")
require_user_management = require_permissions("manage:users")

class UserService:
    """User management service"""
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        # Hash password
        hashed_password = password_manager.hash_password(user_data.password)
        
        # Create user in database
        # This would use the actual database models
        # For now, return a mock response
        user = UserResponse(
            id=1,
            username=user_data.username,
            email=user_data.email,
            role=user_data.role,
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserResponse]:
        """Authenticate user with username/password"""
        # In real implementation, fetch user from database
        # For now, return mock user for valid credentials
        if username == "admin" and password == "admin123":
            return UserResponse(
                id=1,
                username="admin",
                email="admin@example.com",
                role="admin",
                is_active=True,
                created_at=datetime.utcnow(),
                last_login=datetime.utcnow()
            )
        
        return None
    
    def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID"""
        # Mock implementation
        if user_id == 1:
            return UserResponse(
                id=1,
                username="admin",
                email="admin@example.com",
                role="admin",
                is_active=True,
                created_at=datetime.utcnow()
            )
        
        return None

# Global user service instance (would be injected in real app)
user_service = UserService(None)

# Helper functions for API endpoints
def create_user_tokens(user: UserResponse) -> LoginResponse:
    """Create JWT tokens for user"""
    access_token = jwt_manager.create_access_token(
        user_id=user.id,
        username=user.username,
        email=user.email,
        role=user.role
    )
    
    refresh_token = jwt_manager.create_refresh_token(
        user_id=user.id,
        username=user.username
    )
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
        user=user
    )

# Export commonly used items
__all__ = [
    "TokenData",
    "UserCreate", 
    "UserResponse",
    "LoginRequest",
    "LoginResponse",
    "PasswordManager",
    "JWTManager",
    "jwt_manager",
    "password_manager",
    "get_current_user",
    "get_current_active_user", 
    "require_permissions",
    "require_role",
    "require_admin",
    "require_reviewer_or_admin",
    "require_annotator_access",
    "require_export_permission",
    "require_user_management",
    "UserService",
    "user_service",
    "create_user_tokens",
    "ROLE_PERMISSIONS"
]