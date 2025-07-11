# utils/error_handler.py
import traceback
import functools
from typing import Any, Callable, Optional
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MovieFinderError(Exception):
    """Base exception for movie finder errors"""
    pass

class DataLoadError(MovieFinderError):
    """Error loading dataset"""
    pass

class LLMError(MovieFinderError):
    """Error with LLM operations"""
    pass

class FilterError(MovieFinderError):
    """Error with filtering operations"""
    pass

class VectorSearchError(MovieFinderError):
    """Error with vector search operations"""
    pass

def handle_errors(
    default_return: Any = None,
    error_message: str = "An error occurred",
    log_traceback: bool = True
):
    """Decorator for handling errors in functions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{error_message}: {str(e)}"
                logger.error(error_msg)
                
                if log_traceback:
                    logger.error(traceback.format_exc())
                
                return default_return
        return wrapper
    return decorator

def safe_execute(
    func: Callable,
    *args,
    default_return: Any = None,
    error_message: str = "Operation failed",
    **kwargs
) -> Any:
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_msg = f"{error_message}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return default_return

def log_and_return_error(
    error: Exception,
    context: str = "",
    user_friendly_message: str = "An error occurred while processing your request"
) -> str:
    """Log error and return user-friendly message"""
    error_msg = f"❌ Error in {context}: {str(error)}"
    logger.error(error_msg)
    logger.error(traceback.format_exc())
    
    # Return user-friendly message
    return f"❌ {user_friendly_message}. Please try again or contact support if the issue persists."

def validate_dataframe(df, required_columns: list = None) -> bool:
    """Validate DataFrame structure and content"""
    try:
        if df is None or df.empty:
            logger.error("❌ DataFrame is None or empty")
            return False
        
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
        
        logger.info(f"✅ DataFrame validated: {len(df)} rows, {len(df.columns)} columns")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating DataFrame: {str(e)}")
        return False

def validate_llm_response(response) -> bool:
    """Validate LLM response structure"""
    try:
        if response is None:
            logger.error("❌ LLM response is None")
            return False
        
        if hasattr(response, 'content'):
            if not response.content or not response.content.strip():
                logger.error("❌ LLM response content is empty")
                return False
        else:
            if not str(response).strip():
                logger.error("❌ LLM response is empty")
                return False
        
        logger.info("✅ LLM response validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating LLM response: {str(e)}")
        return False

class ErrorContext:
    """Context manager for error handling"""
    
    def __init__(self, operation_name: str, logger_instance=None):
        self.operation_name = operation_name
        self.logger = logger_instance or logger
        self.error_occurred = False
        self.error_message = ""
    
    def __enter__(self):
        self.logger.info(f"🔄 Starting {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_occurred = True
            self.error_message = str(exc_val)
            self.logger.error(f"❌ Error in {self.operation_name}: {exc_val}")
            self.logger.error(traceback.format_exc())
            return False  # Don't suppress the exception
        else:
            self.logger.info(f"✅ Completed {self.operation_name}")
            return True