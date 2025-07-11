
"""
Setup script for Movie Finder System - PHASE 1: Vector Store Creation
Run this FIRST before using the movie finder system
"""

import os
import sys
import time
import traceback
from pathlib import Path
from core.vector_store import VectorStoreManager
from data.imdb_loader import load_dataset
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import pandas as pd
        import numpy as np
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ Core dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Please install required packages:")
        print("   pip install pandas numpy langchain-community langchain-huggingface")
        return False

def check_data_availability():
    """Check if movie data is available"""
    print("📊 Checking data availability...")
    
    try:
        df = load_dataset()
        if df is not None and len(df) > 0:
            print(f"✅ Dataset loaded successfully: {len(df)} movies")
            return True, df
        else:
            print("❌ Dataset is empty or could not be loaded")
            return False, None
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return False, None

def create_vector_store(vector_store_path=None, collection_name=None, force_rebuild=False):
    """
    Phase 1: Create vector store
    
    Args:
        vector_store_path: Path to save vector store (default: './movie_store_vetcor/')
        collection_name: Name of the collection (default: 'imdb_movie')
        force_rebuild: Force rebuild even if vector store exists
    """
    
    # Set default values
    if vector_store_path is None:
        vector_store_path = './movie_store_vetcor/'
    if collection_name is None:
        collection_name = 'imdb_movie'
    
    print("🚀 PHASE 1: Vector Store Creation")
    print("=" * 60)
    print(f"📁 Vector Store Path: {vector_store_path}")
    print(f"🗂️ Collection Name: {collection_name}")
    print(f"🔄 Force Rebuild: {force_rebuild}")
    print("=" * 60)
    
    # Check if vector store already exists
    if os.path.exists(vector_store_path) and not force_rebuild:
        print("⚠️ Vector store already exists!")
        choice = input("Do you want to rebuild it? (y/n): ").lower().strip()
        if choice != 'y':
            print("✅ Using existing vector store")
            return True
    
    try:
        # Load dataset
        print("\n📊 Loading movie dataset...")
        is_data_available, df = check_data_availability()
        if not is_data_available:
            return False
        
        # Create vector store using VectorStoreManager
        print("\n🔧 Creating vector store...")
        start_time = time.time()
        
        VectorStoreManager.create_vector_store(
            vector_store_path=vector_store_path,
            collection_name=collection_name,
            df=df
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ PHASE 1 COMPLETED SUCCESSFULLY!")
        print(f"⏱️ Time taken: {duration:.2f} seconds")
        print(f"📁 Vector store created at: {vector_store_path}")
        print(f"🗂️ Collection: {collection_name}")
        print(f"📊 Movies processed: {len(df)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PHASE 1 FAILED: {str(e)}")
        print("📝 Error details:")
        traceback.print_exc()
        return False

def verify_vector_store(vector_store_path=None, collection_name=None):
    """Verify that the vector store was created successfully"""
    
    if vector_store_path is None:
        vector_store_path = './movie_store_vetcor/'
    if collection_name is None:
        collection_name = 'imdb_movie'
    
    print("\n🔍 Verifying vector store...")
    
    try:
        # Try to load the vector store
        vector_manager = VectorStoreManager(
            vector_store_path=vector_store_path,
            collection_name=collection_name
        )
        
        vector_store = vector_manager.load_existing_vector_store()
        
        # Test with a simple search
        test_results = vector_store.semantic_search("action movies", k=3)
        
        if test_results:
            print(f"✅ Vector store verification successful!")
            print(f"🔍 Test search returned {len(test_results)} results")
            
            # Show sample results
            print("\n📋 Sample search results:")
            for i, result in enumerate(test_results, 1):
                movie_data = result['movie_data']
                title = movie_data.get('Series_Title', 'Unknown')
                year = movie_data.get('Released_Year', 'Unknown')
                print(f"   {i}. {title} ({year})")
            
            return True
        else:
            print("⚠️ Vector store exists but search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Vector store verification failed: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🎬 Movie Finder System Setup - Phase 1")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup failed due to missing dependencies")
        sys.exit(1)
    
    # Get user preferences
    print("\n⚙️ Configuration Options:")
    print("1. Use default settings (recommended)")
    print("2. Custom configuration")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "2":
        vector_store_path = input("Vector store path (default: './movie_store_vetcor/'): ").strip()
        if not vector_store_path:
            vector_store_path = './movie_store_vetcor/'
        
        collection_name = input("Collection name (default: 'imdb_movie'): ").strip()
        if not collection_name:
            collection_name = 'imdb_movie'
        
        force_rebuild = input("Force rebuild existing vector store? (y/n): ").lower().strip() == 'y'
    else:
        vector_store_path = './movie_store_vetcor/'
        collection_name = 'imdb_movie'
        force_rebuild = False
    
    # Create vector store
    success = create_vector_store(
        vector_store_path=vector_store_path,
        collection_name=collection_name,
        force_rebuild=force_rebuild
    )
    
    if success:
        # Verify vector store
        verify_success = verify_vector_store(
            vector_store_path=vector_store_path,
            collection_name=collection_name
        )
        
        if verify_success:
            print("\n🎉 PHASE 1 SETUP COMPLETED SUCCESSFULLY!")
            print("\n📋 Vector Store Details:")
            print(f"   📁 Path: {vector_store_path}")
            print(f"   🗂️ Collection: {collection_name}")
            print(f"   📊 Status: Ready for use")
            print("\n✅ Vector store is now ready for movie processing!")
            
        else:
            print("\n⚠️ Setup completed but verification failed")
            print("Please check the logs for any issues")
    else:
        print("\n❌ SETUP FAILED!")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
