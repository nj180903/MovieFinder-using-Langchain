# core/movie_manager.py
"""
Movie Manager - High-level interface for handling user interactions
Separate from core movie finding logic for better separation of concerns
"""

import os
import traceback
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.movie_finder import EnhancedMovieFinder
from core.vector_store import VectorStoreManager
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SystemStatus(Enum):
    """System status enumeration"""
    NOT_INITIALIZED = "not_initialized"
    VECTOR_STORE_MISSING = "vector_store_missing"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"

@dataclass
class MovieManagerConfig:
    """Configuration for MovieManager"""
    vector_store_path: str = './movie_store_vetcor/'
    collection_name: str = 'imdb_movie'
    enable_logging: bool = True
    max_results: int = 10
    score_threshold: float = 0.5

@dataclass
class QueryResult:
    """Result of a movie query"""
    success: bool
    message: str
    results: Optional[List[Dict]] = None
    query_time: float = 0.0
    search_strategy: str = ""
    error_details: Optional[str] = None

class MovieManager:
    """
    High-level movie manager for handling user interactions
    
    This class provides a clean interface between the UI and the core movie finding logic.
    It handles initialization, validation, error handling, and user session management.
    """
    
    def __init__(self, config: MovieManagerConfig = None):
        """
        Initialize MovieManager
        
        Args:
            config: Configuration for the movie manager
        """
        self.config = config or MovieManagerConfig()
        self.status = SystemStatus.NOT_INITIALIZED
        self.movie_finder: Optional[EnhancedMovieFinder] = None
        self.error_message: Optional[str] = None
        self.initialization_time: Optional[float] = None
        
        logger.info("🎬 MovieManager initialized")
        logger.info(f"📁 Vector store path: {self.config.vector_store_path}")
        logger.info(f"🗂️ Collection name: {self.config.collection_name}")
    
    def check_prerequisites(self) -> Tuple[bool, str]:
        """
        Check if all prerequisites for running the movie manager are met
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if vector store exists
            if not os.path.exists(self.config.vector_store_path):
                return False, f"Vector store not found at: {self.config.vector_store_path}"
            
            # Check if vector store is accessible
            try:
                vector_manager = VectorStoreManager(
                    vector_store_path=self.config.vector_store_path,
                    collection_name=self.config.collection_name
                )
                # Try to load to verify it's valid
                vector_manager.load_existing_vector_store()
                return True, "All prerequisites met"
                
            except Exception as e:
                return False, f"Vector store exists but cannot be loaded: {str(e)}"
                
        except Exception as e:
            return False, f"Error checking prerequisites: {str(e)}"
    
    def initialize(self) -> bool:
        """
        Initialize the movie manager system
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.status = SystemStatus.INITIALIZING
            self.error_message = None
            
            logger.info("🚀 Initializing MovieManager system...")
            
            # Check prerequisites
            prereq_ok, prereq_msg = self.check_prerequisites()
            if not prereq_ok:
                self.status = SystemStatus.VECTOR_STORE_MISSING
                self.error_message = prereq_msg
                logger.error(f"❌ Prerequisites not met: {prereq_msg}")
                return False
            
            # Initialize movie finder
            import time
            start_time = time.time()
            
            self.movie_finder = EnhancedMovieFinder(
                vector_store_path=self.config.vector_store_path,
                collection_name=self.config.collection_name
            )
            
            # Initialize the movie finder system
            self.movie_finder._initialize_system()
            
            self.initialization_time = time.time() - start_time
            self.status = SystemStatus.READY
            
            logger.info(f"✅ MovieManager initialized successfully in {self.initialization_time:.2f}s")
            return True
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.error_message = str(e)
            logger.error(f"❌ MovieManager initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def process_user_query(self, query: str) -> QueryResult:
        """
        Process a user movie query
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryResult with results and metadata
        """
        import time
        start_time = time.time()
        
        # Validate input
        if not query or not query.strip():
            return QueryResult(
                success=False,
                message="Please enter a valid movie query.",
                query_time=time.time() - start_time
            )
        
        # Check system status
        if self.status != SystemStatus.READY:
            return QueryResult(
                success=False,
                message=f"System not ready. Status: {self.status.value}",
                error_details=self.error_message,
                query_time=time.time() - start_time
            )
        
        try:
            logger.info(f"🔍 Processing user query: {query}")
            
            # Determine search strategy before processing
            search_strategy = self.movie_finder._determine_search_strategy(query)
            
            # Process the query
            result_text = self.movie_finder.process_query(query)
            
            query_time = time.time() - start_time
            
            # Check if result indicates an error
            if result_text.startswith("❌"):
                return QueryResult(
                    success=False,
                    message=result_text,
                    search_strategy=search_strategy,
                    query_time=query_time,
                    error_details="Query processing error"
                )
            
            # Check if no results found
            if result_text.startswith("😕"):
                return QueryResult(
                    success=True,
                    message=result_text,
                    search_strategy=search_strategy,
                    query_time=query_time,
                    results=[]
                )
            
            # Successful result
            return QueryResult(
                success=True,
                message=result_text,
                search_strategy=search_strategy,
                query_time=query_time
            )
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            return QueryResult(
                success=False,
                message=error_msg,
                error_details=str(e),
                query_time=time.time() - start_time
            )
    
    def get_system_status(self) -> Dict:
        """
        Get comprehensive system status information
        
        Returns:
            Dictionary with system status details
        """
        status_info = {
            "status": self.status.value,
            "vector_store_path": self.config.vector_store_path,
            "collection_name": self.config.collection_name,
            "error_message": self.error_message,
            "initialization_time": self.initialization_time,
            "movie_finder_ready": self.movie_finder is not None
        }
        
        # Add movie finder system info if available
        if self.movie_finder:
            try:
                movie_finder_info = self.movie_finder.get_system_info()
                status_info["movie_finder_details"] = movie_finder_info
            except Exception as e:
                status_info["movie_finder_error"] = str(e)
        
        return status_info
    
    def get_conversation_history(self) -> List[Dict]:
        """
        Get conversation history
        
        Returns:
            List of conversation messages
        """
        if self.movie_finder and self.movie_finder.history:
            return self.movie_finder.history.messages
        return []
    
    def clear_conversation_history(self) -> bool:
        """
        Clear conversation history
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.movie_finder and self.movie_finder.history:
                self.movie_finder.clear_history()
                logger.info("✅ Conversation history cleared")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Error clearing history: {str(e)}")
            return False
    
    def get_search_suggestions(self) -> List[str]:
        """
        Get search query suggestions
        
        Returns:
            List of example queries
        """
        return [
            "Find action movies with high ratings",
            "Show me comedies from the 1990s", 
            "Movies similar to Inception",
            "Christopher Nolan thriller movies",
            "Sci-fi movies about AI",
            "Best drama movies of all time",
            "Tom Cruise action movies",
            "Movies with time travel theme",
            "Horror movies from the 1980s",
            "Romantic comedies with Julia Roberts"
        ]
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate a user query before processing
        
        Args:
            query: User query to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not query:
            return False, "Query cannot be empty"
        
        if not query.strip():
            return False, "Query cannot be just whitespace"
        
        if len(query.strip()) < 3:
            return False, "Query too short - please provide more details"
        
        if len(query) > 1000:
            return False, "Query too long - please keep it under 1000 characters"
        
        return True, "Query is valid"
    
    def restart_system(self) -> bool:
        """
        Restart the movie manager system
        
        Returns:
            True if restart successful, False otherwise
        """
        try:
            logger.info("🔄 Restarting MovieManager system...")
            
            # Clear current state
            self.movie_finder = None
            self.error_message = None
            self.initialization_time = None
            self.status = SystemStatus.NOT_INITIALIZED
            
            # Reinitialize
            return self.initialize()
            
        except Exception as e:
            logger.error(f"❌ Error restarting system: {str(e)}")
            self.status = SystemStatus.ERROR
            self.error_message = str(e)
            return False

# Singleton pattern for global movie manager instance
_global_movie_manager: Optional[MovieManager] = None

def get_movie_manager(config: MovieManagerConfig = None) -> MovieManager:
    """
    Get the global movie manager instance
    
    Args:
        config: Configuration (only used for first initialization)
        
    Returns:
        MovieManager instance
    """
    global _global_movie_manager
    
    if _global_movie_manager is None:
        _global_movie_manager = MovieManager(config)
    
    return _global_movie_manager

def reset_movie_manager():
    """Reset the global movie manager instance"""
    global _global_movie_manager
    _global_movie_manager = None 