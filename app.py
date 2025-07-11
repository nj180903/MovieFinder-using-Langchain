import streamlit as st
import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.movie_manager import MovieManager, MovieManagerConfig

# Configuration
DEFAULT_CONFIG = MovieManagerConfig(
    vector_store_path='./movie_store_vetcor/',
    collection_name='imdb_movie',
    enable_logging=True,
    max_results=10,
    score_threshold=0.5
)

def main():
    """Simple Streamlit app with input and output"""
    # Set page config
    st.set_page_config(
        page_title="Movie Finder - Simple Interface",
        page_icon="🎬",
        layout="wide"
    )
    
    # Header
    st.title("🎬 Movie Finder - Simple Interface")
    st.markdown("Enter your movie query and get recommendations")
    
    # Input section
    st.header("📝 Input")
    user_query = st.text_area(
        "Enter your movie query:",
        placeholder="e.g., 'Find me action movies with Tom Cruise'",
        height=100
    )
    
    # Process button
    if st.button("🔍 Search Movies", type="primary"):
        if user_query:
            # Output section
            st.header("📋 Output")
            
            with st.spinner("Processing your query..."):
                try:
                    # Initialize movie manager
                    movie_manager = MovieManager(DEFAULT_CONFIG)
                    
                    # Check if system is ready
                    if not movie_manager.check_prerequisites()[0]:
                        st.error("❌ System not ready. Please run setup.py first!")
                        st.info("Run: `python setup.py` in the pls directory")
                        return
                    
                    # Initialize the system
                    if movie_manager.initialize():
                        # Process the query
                        result = movie_manager.process_user_query(user_query)
                        
                        if result.success:
                            st.success("✅ Search completed successfully!")
                            st.markdown("### 🎬 Movie Recommendations:")
                            st.markdown(result.message)
                            
                            # Show some metadata
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("⏱️ Query Time", f"{result.query_time:.2f}s")
                            with col2:
                                st.metric("🎯 Strategy", result.search_strategy.title())
                        else:
                            st.error(f"❌ Search failed: {result.message}")
                            if result.error_details:
                                st.error(f"Details: {result.error_details}")
                    else:
                        st.error("❌ Failed to initialize the movie system")
                        
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.error("Please make sure you have run setup.py first!")
        else:
            st.warning("⚠️ Please enter a movie query")

if __name__ == "__main__":
    main()
