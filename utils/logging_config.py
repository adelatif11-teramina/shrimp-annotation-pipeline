"""
Secure Logging Configuration
Filters sensitive data and provides structured logging
"""

import os
import re
import json
import logging
import logging.config
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys

# Sensitive data patterns to filter out
SENSITIVE_PATTERNS = [
    # API Keys
    r'(?i)(api[_-]?key|apikey)[\s]*[:=][\s]*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    r'(?i)(secret[_-]?key|secretkey)[\s]*[:=][\s]*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    r'(?i)(access[_-]?token|accesstoken)[\s]*[:=][\s]*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
    r'(?i)(bearer[\s]+)([a-zA-Z0-9_\-\.]{20,})',
    
    # OpenAI API keys
    r'sk-[a-zA-Z0-9]{48}',
    r'sk-proj-[a-zA-Z0-9_\-]{64,}',
    
    # Database URLs with passwords
    r'(postgresql|mysql|mongodb)://[^:]+:([^@]+)@',
    r'(?i)(password|pwd)[\s]*[:=][\s]*["\']?([^"\'\s]{8,})["\']?',
    
    # JWT tokens
    r'eyJ[a-zA-Z0-9_\-]*\.eyJ[a-zA-Z0-9_\-]*\.[a-zA-Z0-9_\-]*',
    
    # Credit card numbers
    r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
    
    # Email addresses (partial masking)
    r'\b([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b',
    
    # IP addresses (may be sensitive in some contexts)
    r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
]

class SensitiveDataFilter:
    """Filter to remove sensitive data from log messages"""
    
    def __init__(self, patterns: Optional[List[str]] = None):
        self.patterns = patterns or SENSITIVE_PATTERNS
        self.compiled_patterns = [re.compile(pattern) for pattern in self.patterns]
    
    def filter_message(self, message: str) -> str:
        """Remove sensitive data from a message"""
        filtered_message = message
        
        for pattern in self.compiled_patterns:
            if pattern.search(filtered_message):
                # Different replacement strategies based on pattern
                if 'email' in pattern.pattern.lower():
                    # Partial email masking: user***@domain.com
                    filtered_message = pattern.sub(r'\1***@\2', filtered_message)
                elif 'ip' in pattern.pattern.lower():
                    # IP masking: 192.168.1.***
                    filtered_message = pattern.sub(r'***.***.***.***', filtered_message)
                elif 'bearer' in pattern.pattern.lower():
                    # Bearer token masking
                    filtered_message = pattern.sub(r'\1***', filtered_message)
                elif 'postgresql' in pattern.pattern.lower() or 'mysql' in pattern.pattern.lower():
                    # Database URL password masking
                    filtered_message = pattern.sub(r'\1://***:***@', filtered_message)
                else:
                    # General sensitive data masking
                    filtered_message = pattern.sub('***REDACTED***', filtered_message)
        
        return filtered_message

class SecureLogFilter(logging.Filter):
    """Logging filter that removes sensitive data"""
    
    def __init__(self):
        super().__init__()
        self.sensitive_filter = SensitiveDataFilter()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log record to remove sensitive data"""
        # Filter the main message
        if hasattr(record, 'msg') and record.msg:
            record.msg = self.sensitive_filter.filter_message(str(record.msg))
        
        # Filter any arguments
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    filtered_args.append(self.sensitive_filter.filter_message(arg))
                else:
                    filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        # Filter extra fields
        for key, value in record.__dict__.items():
            if isinstance(value, str) and key not in ['name', 'levelname', 'funcName', 'module']:
                setattr(record, key, self.sensitive_filter.filter_message(value))
        
        return True

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if enabled
        if self.include_extra:
            # Standard fields to exclude
            exclude_fields = {
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                'thread', 'threadName', 'processName', 'process', 'exc_info', 'exc_text',
                'stack_info'
            }
            
            # Add any extra fields
            for key, value in record.__dict__.items():
                if key not in exclude_fields:
                    log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)

class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def __init__(self):
        super().__init__()
        self.start_time = datetime.utcnow()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance info to record"""
        record.uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return True

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    enable_sensitive_filter: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup application logging with security filters
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, logs to console only)
        json_format: Use JSON structured logging
        enable_sensitive_filter: Enable sensitive data filtering
        max_bytes: Maximum bytes per log file
        backup_count: Number of backup files to keep
    """
    
    # Normalize level
    if isinstance(level, str):
        level = level.upper()

    # Create logs directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Base configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'filters': {},
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': sys.stdout
            }
        },
        'loggers': {
            # Application loggers
            'services': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            'utils': {
                'level': level,
                'handlers': ['console'], 
                'propagate': False
            },
            'config': {
                'level': level,
                'handlers': ['console'],
                'propagate': False
            },
            # Third-party loggers (less verbose)
            'uvicorn': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'fastapi': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'sqlalchemy': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'alembic': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'openai': {
                'level': 'WARNING',
                'handlers': ['console'],
                'propagate': False
            },
            'httpx': {
                'level': 'WARNING', 
                'handlers': ['console'],
                'propagate': False
            }
        },
        'root': {
            'level': level,
            'handlers': ['console']
        }
    }
    
    # Add JSON formatter if requested
    if json_format:
        config['formatters']['json'] = {
            '()': StructuredFormatter,
            'include_extra': True
        }
        config['handlers']['console']['formatter'] = 'json'
    
    # Add sensitive data filter if enabled
    if enable_sensitive_filter:
        config['filters']['sensitive'] = {
            '()': SecureLogFilter
        }
        config['handlers']['console']['filters'] = ['sensitive']
    
    # Add performance filter
    config['filters']['performance'] = {
        '()': PerformanceFilter
    }
    if 'filters' not in config['handlers']['console']:
        config['handlers']['console']['filters'] = []
    config['handlers']['console']['filters'].append('performance')
    
    # Add file handler if log file specified
    if log_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'json' if json_format else 'detailed',
            'filename': log_file,
            'maxBytes': max_bytes,
            'backupCount': backup_count,
            'encoding': 'utf-8'
        }
        
        if enable_sensitive_filter:
            config['handlers']['file']['filters'] = ['sensitive', 'performance']
        else:
            config['handlers']['file']['filters'] = ['performance']
        
        # Add file handler to all loggers
        for logger_config in config['loggers'].values():
            if 'file' not in logger_config['handlers']:
                logger_config['handlers'].append('file')
        
        config['root']['handlers'].append('file')
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Test the configuration
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully")
    logger.info(f"Log level: {level}")
    logger.info(f"JSON format: {json_format}")
    logger.info(f"Sensitive data filtering: {enable_sensitive_filter}")
    if log_file:
        logger.info(f"File logging: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    return logging.getLogger(name)

def log_function_call(func_name: str, args: Dict[str, Any], result: Any = None, error: Exception = None):
    """Log a function call with arguments and result"""
    logger = logging.getLogger(func_name)
    
    # Filter sensitive data from arguments
    sensitive_filter = SensitiveDataFilter()
    filtered_args = {}
    
    for key, value in args.items():
        if isinstance(value, str):
            filtered_args[key] = sensitive_filter.filter_message(value)
        else:
            filtered_args[key] = value
    
    if error:
        logger.error(f"Function {func_name} failed", extra={
            'function': func_name,
            'arguments': filtered_args,
            'error': str(error),
            'error_type': type(error).__name__
        })
    else:
        logger.info(f"Function {func_name} completed", extra={
            'function': func_name,
            'arguments': filtered_args,
            'has_result': result is not None
        })

def log_api_request(method: str, path: str, status_code: int, duration: float, user_id: Optional[str] = None):
    """Log an API request"""
    logger = logging.getLogger('api.requests')
    
    log_data = {
        'method': method,
        'path': path,
        'status_code': status_code,
        'duration_ms': round(duration * 1000, 2),
        'user_id': user_id
    }
    
    if status_code >= 500:
        logger.error("API request failed", extra=log_data)
    elif status_code >= 400:
        logger.warning("API request error", extra=log_data)
    else:
        logger.info("API request completed", extra=log_data)

def log_database_operation(operation: str, table: str, duration: float, rows_affected: Optional[int] = None):
    """Log a database operation"""
    logger = logging.getLogger('database.operations')
    
    log_data = {
        'operation': operation,
        'table': table,
        'duration_ms': round(duration * 1000, 2)
    }
    
    if rows_affected is not None:
        log_data['rows_affected'] = rows_affected
    
    if duration > 1.0:  # Slow query threshold
        logger.warning("Slow database operation", extra=log_data)
    else:
        logger.debug("Database operation completed", extra=log_data)

def log_llm_request(provider: str, model: str, tokens_used: Optional[int] = None, duration: float = 0.0, error: Optional[str] = None):
    """Log an LLM API request"""
    logger = logging.getLogger('llm.requests')
    
    log_data = {
        'provider': provider,
        'model': model,
        'duration_ms': round(duration * 1000, 2)
    }
    
    if tokens_used:
        log_data['tokens_used'] = tokens_used
    
    if error:
        log_data['error'] = error
        logger.error("LLM request failed", extra=log_data)
    else:
        logger.info("LLM request completed", extra=log_data)

# Context manager for operation logging
class LogOperation:
    """Context manager for logging operations with timing"""
    
    def __init__(self, operation_name: str, logger_name: str = 'operations', extra_data: Optional[Dict[str, Any]] = None):
        self.operation_name = operation_name
        self.logger = logging.getLogger(logger_name)
        self.extra_data = extra_data or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting {self.operation_name}", extra={
            'operation': self.operation_name,
            'status': 'started',
            **self.extra_data
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        log_data = {
            'operation': self.operation_name,
            'duration_ms': round(duration * 1000, 2),
            **self.extra_data
        }
        
        if exc_type:
            log_data['status'] = 'failed'
            log_data['error'] = str(exc_val)
            log_data['error_type'] = exc_type.__name__
            self.logger.error(f"Operation {self.operation_name} failed", extra=log_data)
        else:
            log_data['status'] = 'completed'
            self.logger.info(f"Operation {self.operation_name} completed", extra=log_data)

# Export commonly used items
__all__ = [
    'setup_logging',
    'get_logger',
    'log_function_call',
    'log_api_request', 
    'log_database_operation',
    'log_llm_request',
    'LogOperation',
    'SensitiveDataFilter',
    'SecureLogFilter',
    'StructuredFormatter'
]
