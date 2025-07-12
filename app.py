

import streamlit as st
import os
import sys
from pathlib import Path
import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3
# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.movie_manager import MovieManager, MovieManagerConfig

# Try importing setup script
try:
    from setup import create_vector_store, verify_vector_store
except ImportError:
    st.error("❌ setup.py not found or invalid!")
    st.stop()

# Configuration
VECTOR_STORE_PATH = './movie_store_vetcor/'
COLLECTION_NAME = 'imdb_movie'

DEFAULT_CONFIG = MovieManagerConfig(
    vector_store_path=VECTOR_STORE_PATH,
    collection_name=COLLECTION_NAME,
    enable_logging=True,
    max_results=10,
    score_threshold=0.5
)

def auto_run_setup():
    """Automatically run setup phase if vector store is missing"""
    if not os.path.exists(VECTOR_STORE_PATH):
        st.warning("⚠️ Vector store not found. Running setup now...")
        with st.spinner("🔧 Setting up vector store..."):
            success = create_vector_store(
                vector_store_path=VECTOR_STORE_PATH,
                collection_name=COLLECTION_NAME,
                force_rebuild=False
            )
            if success:
                verified = verify_vector_store(
                    vector_store_path=VECTOR_STORE_PATH,
                    collection_name=COLLECTION_NAME
                )
                if verified:
                    st.success("✅ Setup completed and verified successfully!")
                else:
                    st.error("❌ Vector store verification failed.")
                    st.stop()
            else:
                st.error("❌ Setup failed. Check setup.py or logs.")
                st.stop()

def main():
    """Streamlit app"""
    st.set_page_config(
        page_title="Movie Finder - Simple Interface",
        page_icon="🎬",
        layout="wide"
    )

    st.title("🎬 Movie Finder - Simple Interface")
    st.markdown("Enter your movie query and get recommendations")

    # Run setup if needed
    auto_run_setup()

    # Input section
    st.header("📝 Input")
    user_query = st.text_area(
        "Enter your movie query:",
        placeholder="e.g., 'Find me action movies with Tom Cruise'",
        height=100
    )

    if st.button("🔍 Search Movies", type="primary"):
        if user_query:
            st.header("📋 Output")
            with st.spinner("Processing your query..."):
                try:
                    movie_manager = MovieManager(DEFAULT_CONFIG)
                    
                    if movie_manager.initialize():
                        result = movie_manager.process_user_query(user_query)
                        
                        if result.success:
                            st.success("✅ Search completed successfully!")
                            st.markdown("### 🎬 Movie Recommendations:")
                            st.markdown(result.message)
                            
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
        else:
            st.warning("⚠️ Please enter a movie query")

if __name__ == "__main__":
    main()
