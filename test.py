# main.py
import gradio as gr
import traceback
from core.movie_finder import EnhancedMovieFinder
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Global movie finder instance
movie_finder = None

def initialize_system():
    """Initialize the movie finder system"""
    global movie_finder
    try:
        logger.info("🚀 Initializing Movie Finder System...")
        movie_finder = EnhancedMovieFinder()
        movie_finder._initialize_system()
        return "✅ System initialized successfully!"
    except Exception as e:
        error_msg = f"❌ Failed to initialize system: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg

def process_query(query: str) -> str:
    """Process movie query with error handling"""
    global movie_finder
    
    if movie_finder is None:
        return "❌ System not initialized. Please wait for initialization to complete."
    
    if not query.strip():
        return "Please enter a movie query."
    
    return movie_finder.process_query(query)



# if you want quick tst without gradaio, uncomment this
# ini
# if movie_finder is None:
#     initialize_system()

# 
# process_query("hi can u help me to choose top 5 romantic movies")