"""
Rate Limiting and API Protection
"""

import time
import asyncio
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import wraps
import hashlib
import redis
from fastapi import Request, HTTPException, Depends
from starlette.responses import JSONResponse

@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    requests: int  # Number of requests
    window: int   # Time window in seconds
    key_func: Optional[Callable[[Request], str]] = None  # Custom key function

class RateLimiter:
    """Thread-safe rate limiter with multiple algorithms"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_cache: Dict[str, deque] = defaultdict(lambda: deque())
        self.local_lock = asyncio.Lock()
    
    async def is_allowed(
        self, 
        key: str, 
        limit: int, 
        window: int,
        algorithm: str = "sliding_window"
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Unique identifier for rate limit (e.g., user ID, IP)
            limit: Number of requests allowed
            window: Time window in seconds
            algorithm: Rate limiting algorithm
            
        Returns:
            (is_allowed, info_dict)
        """
        if self.redis_client:
            return await self._redis_check(key, limit, window, algorithm)
        else:
            return await self._memory_check(key, limit, window, algorithm)
    
    async def _redis_check(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        algorithm: str
    ) -> tuple[bool, Dict[str, Any]]:
        """Redis-based rate limiting"""
        now = time.time()
        redis_key = f"rate_limit:{key}"
        
        if algorithm == "sliding_window":
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, now - window)
            
            # Count current requests
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(now): now})
            
            # Set expiry
            pipe.expire(redis_key, window + 1)
            
            results = pipe.execute()
            current_requests = results[1]
            
            # Check if over limit
            is_allowed = current_requests < limit
            
            if not is_allowed:
                # Remove the request we just added
                self.redis_client.zrem(redis_key, str(now))
            
            # Get reset time
            oldest_request = self.redis_client.zrange(redis_key, 0, 0, withscores=True)
            reset_time = int(oldest_request[0][1] + window) if oldest_request else int(now + window)
            
            return is_allowed, {
                "limit": limit,
                "remaining": max(0, limit - current_requests - (0 if is_allowed else 1)),
                "reset": reset_time,
                "retry_after": reset_time - int(now) if not is_allowed else 0
            }
            
        elif algorithm == "token_bucket":
            # Token bucket using Redis hash
            bucket_key = f"bucket:{key}"
            
            # Get current bucket state
            bucket_data = self.redis_client.hmget(bucket_key, ["tokens", "last_refill"])
            
            tokens = float(bucket_data[0] or limit)
            last_refill = float(bucket_data[1] or now)
            
            # Calculate tokens to add based on elapsed time
            time_passed = now - last_refill
            tokens_to_add = time_passed * (limit / window)  # Refill rate
            tokens = min(limit, tokens + tokens_to_add)
            
            # Check if request is allowed
            is_allowed = tokens >= 1
            
            if is_allowed:
                tokens -= 1
            
            # Update bucket state
            self.redis_client.hmset(bucket_key, {
                "tokens": tokens,
                "last_refill": now
            })
            self.redis_client.expire(bucket_key, window * 2)
            
            return is_allowed, {
                "limit": limit,
                "remaining": int(tokens),
                "reset": int(now + (limit - tokens) * (window / limit)),
                "retry_after": int((1 - tokens) * (window / limit)) if not is_allowed else 0
            }
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    async def _memory_check(
        self, 
        key: str, 
        limit: int, 
        window: int, 
        algorithm: str
    ) -> tuple[bool, Dict[str, Any]]:
        """Memory-based rate limiting (for development/testing)"""
        async with self.local_lock:
            now = time.time()
            
            if algorithm == "sliding_window":
                requests = self.local_cache[key]
                
                # Remove expired requests
                while requests and requests[0] <= now - window:
                    requests.popleft()
                
                # Check if over limit
                is_allowed = len(requests) < limit
                
                if is_allowed:
                    requests.append(now)
                
                # Calculate reset time
                reset_time = int(requests[0] + window) if requests else int(now + window)
                
                return is_allowed, {
                    "limit": limit,
                    "remaining": max(0, limit - len(requests)),
                    "reset": reset_time,
                    "retry_after": reset_time - int(now) if not is_allowed else 0
                }
            
            elif algorithm == "token_bucket":
                # Simplified token bucket for memory storage
                bucket_data = self.local_cache.get(f"bucket:{key}", [limit, now])
                tokens, last_refill = bucket_data
                
                # Refill tokens
                time_passed = now - last_refill
                tokens_to_add = time_passed * (limit / window)
                tokens = min(limit, tokens + tokens_to_add)
                
                # Check if allowed
                is_allowed = tokens >= 1
                
                if is_allowed:
                    tokens -= 1
                
                # Update bucket
                self.local_cache[f"bucket:{key}"] = [tokens, now]
                
                return is_allowed, {
                    "limit": limit,
                    "remaining": int(tokens),
                    "reset": int(now + (limit - tokens) * (window / limit)),
                    "retry_after": int((1 - tokens) * (window / limit)) if not is_allowed else 0
                }
            
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, rate_limiter: RateLimiter, rules: Dict[str, RateLimitRule]):
        self.rate_limiter = rate_limiter
        self.rules = rules
    
    def get_key(self, request: Request, rule: RateLimitRule) -> str:
        """Generate rate limit key for request"""
        if rule.key_func:
            return rule.key_func(request)
        
        # Default: use IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting"""
        # Check if endpoint has rate limiting
        endpoint = f"{request.method}:{request.url.path}"
        
        # Apply global rules first
        if "global" in self.rules:
            rule = self.rules["global"]
            key = self.get_key(request, rule)
            
            is_allowed, info = await self.rate_limiter.is_allowed(
                key, rule.requests, rule.window
            )
            
            if not is_allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": info["retry_after"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )
        
        # Check endpoint-specific rules
        if endpoint in self.rules:
            rule = self.rules[endpoint]
            key = self.get_key(request, rule)
            
            is_allowed, info = await self.rate_limiter.is_allowed(
                key, rule.requests, rule.window
            )
            
            if not is_allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": f"Rate limit exceeded for {endpoint}",
                        "retry_after": info["retry_after"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": str(info["remaining"]),
                        "X-RateLimit-Reset": str(info["reset"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        if "global" in self.rules:
            rule = self.rules["global"]
            key = self.get_key(request, rule)
            _, info = await self.rate_limiter.is_allowed(key, rule.requests, rule.window)
            
            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])
        
        return response

def rate_limit(
    requests: int, 
    window: int, 
    per: str = "ip",
    algorithm: str = "sliding_window"
):
    """
    Decorator for endpoint-specific rate limiting
    
    Args:
        requests: Number of requests allowed
        window: Time window in seconds  
        per: Rate limit key ("ip", "user", or custom function)
        algorithm: Rate limiting algorithm
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This is handled by middleware in practice
            # This decorator is mainly for documentation
            return await func(*args, **kwargs)
        
        # Attach rate limit metadata
        wrapper._rate_limit = {
            "requests": requests,
            "window": window,
            "per": per,
            "algorithm": algorithm
        }
        
        return wrapper
    return decorator

def get_user_id(request: Request) -> str:
    """Extract user ID from request for user-based rate limiting"""
    # Try to get user from JWT token or session
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # In real implementation, decode JWT and extract user ID
        # For now, use token hash as user identifier
        return f"user:{hashlib.md5(token.encode()).hexdigest()[:8]}"
    
    # Fallback to IP-based
    client_ip = request.client.host
    return f"anonymous:{client_ip}"

def get_api_key(request: Request) -> str:
    """Extract API key for API key-based rate limiting"""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"
    
    # Fallback to IP-based
    return f"no_api_key:{request.client.host}"

# Rate limiting configurations for different environments
RATE_LIMIT_CONFIGS = {
    "development": {
        "global": RateLimitRule(requests=1000, window=60),  # 1000/minute
        "POST:/candidates/generate": RateLimitRule(requests=50, window=60),  # 50/minute
        "POST:/candidates/batch": RateLimitRule(requests=10, window=60),  # 10/minute
        "POST:/documents": RateLimitRule(requests=100, window=60),  # 100/minute
    },
    "production": {
        "global": RateLimitRule(requests=100, window=60, key_func=get_user_id),  # 100/minute per user
        "POST:/candidates/generate": RateLimitRule(requests=20, window=60, key_func=get_user_id),  # 20/minute per user
        "POST:/candidates/batch": RateLimitRule(requests=5, window=60, key_func=get_user_id),  # 5/minute per user
        "POST:/documents": RateLimitRule(requests=50, window=60, key_func=get_user_id),  # 50/minute per user
        "api_endpoints": RateLimitRule(requests=1000, window=60, key_func=get_api_key),  # API key rate limits
    }
}

def create_rate_limiter(redis_url: Optional[str] = None) -> RateLimiter:
    """Create rate limiter instance"""
    redis_client = None
    
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            # Test connection
            redis_client.ping()
        except Exception as e:
            print(f"Failed to connect to Redis: {e}")
            print("Falling back to memory-based rate limiting")
            redis_client = None
    
    return RateLimiter(redis_client)

# Export commonly used items
__all__ = [
    "RateLimiter",
    "RateLimitRule", 
    "RateLimitMiddleware",
    "rate_limit",
    "get_user_id",
    "get_api_key",
    "RATE_LIMIT_CONFIGS",
    "create_rate_limiter"
]