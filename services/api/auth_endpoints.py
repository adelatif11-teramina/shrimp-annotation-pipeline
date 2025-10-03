"""
Authentication API Endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPAuthorizationCredentials
from utils.auth import (
    LoginRequest, LoginResponse, UserCreate, UserResponse, TokenData,
    jwt_manager, user_service, create_user_tokens, get_current_user,
    require_admin, require_user_management
)

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/login", response_model=LoginResponse)
async def login(login_request: LoginRequest):
    """
    Authenticate user and return JWT tokens
    """
    try:
        # Authenticate user
        user = user_service.authenticate_user(
            login_request.username,
            login_request.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user account",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        tokens = create_user_tokens(user)
        
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

@router.post("/refresh", response_model=dict)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        payload = jwt_manager.verify_refresh_token(refresh_token)
        
        # Get user data
        user = user_service.get_user_by_id(payload["user_id"])
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new access token
        new_access_token = jwt_manager.create_access_token(
            user_id=user.id,
            username=user.username,
            email=user.email,
            role=user.role
        )
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": jwt_manager.access_token_expire_minutes * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout")
async def logout(current_user: TokenData = Depends(get_current_user)):
    """
    Logout user (revoke current token)
    """
    try:
        # In a real implementation, you'd revoke the token
        # For now, just return success
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: TokenData = Depends(get_current_user)):
    """
    Get current user information
    """
    try:
        user = user_service.get_user_by_id(current_user.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: TokenData = Depends(require_user_management)
):
    """
    Create a new user (admin only)
    """
    try:
        # Check if username already exists (in real implementation)
        # For now, just create the user
        
        new_user = user_service.create_user(user_data)
        return new_user
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: TokenData = Depends(require_admin),
    skip: int = 0,
    limit: int = 100
):
    """
    List all users (admin only)
    """
    try:
        # In real implementation, fetch from database
        # For now, return mock data
        users = [
            UserResponse(
                id=1,
                username="admin",
                email="admin@example.com",
                role="admin",
                is_active=True,
                created_at="2024-01-01T00:00:00"
            ),
            UserResponse(
                id=2,
                username="annotator1",
                email="annotator1@example.com",
                role="annotator",
                is_active=True,
                created_at="2024-01-02T00:00:00"
            )
        ]
        
        return users[skip:skip + limit]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

@router.get("/verify-token")
async def verify_token(current_user: TokenData = Depends(get_current_user)):
    """
    Verify if current token is valid
    """
    return {
        "valid": True,
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role,
        "permissions": current_user.permissions,
        "expires_at": current_user.exp.isoformat()
    }

@router.get("/permissions")
async def get_user_permissions(current_user: TokenData = Depends(get_current_user)):
    """
    Get current user's permissions
    """
    return {
        "user_id": current_user.user_id,
        "role": current_user.role,
        "permissions": current_user.permissions
    }