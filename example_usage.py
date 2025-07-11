# example_usage.py
"""
Example usage of the refactored MovieFinder system with strict two-phase approach:
PHASE 1: Vector store creation (mandatory first step)
PHASE 2: Movie processing (uses existing vector store only)
"""

import os
from core.movie_finder import EnhancedMovieFinder, VectorStoreManager
from data.imdb_loader import load_dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)

def phase_1_create_vector_store():
    """PHASE 1: Create vector store - This MUST be done first"""
    print("🚀 PHASE 1: Creating Vector Store")
    print("=" * 60)
    
    # Method 1: Using static method (recommended)
    print("\n📦 Method 1: Using VectorStoreManager.create_vector_store()")
    VectorStoreManager.create_vector_store(
        vector_store_path="./movie_vectors",
        collection_name="imdb_movies"
    )
    
    print("\n📦 Method 2: Using VectorStoreManager instance")
    # Alternative: Using VectorStoreManager instance
    vector_manager = VectorStoreManager(
        vector_store_path="./custom_movie_vectors",
        collection_name="custom_movies"
    )
    
    # Load dataset and build vector store
    df = load_dataset()
    vector_manager.build_vector_store(df)
    
    print("\n✅ PHASE 1 COMPLETED: Vector stores created successfully")
    print("=" * 60)

def phase_2_movie_processing():
    """PHASE 2: Movie processing - Uses existing vector store only"""
    print("\n🎬 PHASE 2: Movie Processing")
    print("=" * 60)
    
    try:
        # Initialize MovieFinder with existing vector store
        movie_finder = EnhancedMovieFinder(
            vector_store_path="./movie_vectors",
            collection_name="imdb_movies"
        )
        
        # Initialize the system for processing
        movie_finder._initialize_system()
        
        # Process queries
        queries = [
            "Find me some action movies with high ratings",
            "Show me comedies from the 1990s",
            "Find sci-fi movies similar to Blade Runner",
            "What are some good horror movies?"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            print("-" * 40)
            response = movie_finder.process_query(query)
            print(f"📝 Response: {response}")
            print("-" * 40)
        
        # Get system info
        print("\n📊 System Information:")
        print(movie_finder.get_system_info())
        
    except Exception as e:
        print(f"❌ Error in Phase 2: {str(e)}")
        print("🚨 Make sure you ran Phase 1 first!")

def example_complete_workflow():
    """Example of complete workflow: Phase 1 + Phase 2"""
    print("\n🔄 COMPLETE WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # PHASE 1: Create vector store
    print("\n🚀 Starting Phase 1...")
    VectorStoreManager.create_vector_store(
        vector_store_path="./workflow_vectors",
        collection_name="workflow_movies"
    )
    
    # PHASE 2: Process movies
    print("\n🎬 Starting Phase 2...")
    movie_finder = EnhancedMovieFinder(
        vector_store_path="./workflow_vectors",
        collection_name="workflow_movies"
    )
    
    movie_finder._initialize_system()
    
    # Process some queries
    response = movie_finder.process_query("Find me thriller movies with great plots")
    print(f"\n📝 Response: {response}")
    
    print("\n✅ Complete workflow finished successfully!")

def example_error_handling():
    """Example showing what happens when Phase 1 is skipped"""
    print("\n⚠️ ERROR HANDLING EXAMPLE")
    print("=" * 60)
    
    try:
        # Try to initialize MovieFinder without creating vector store first
        movie_finder = EnhancedMovieFinder(
            vector_store_path="./non_existent_vectors",
            collection_name="non_existent_collection"
        )
        
        # This will fail because vector store doesn't exist
        movie_finder._initialize_system()
        
    except Exception as e:
        print(f"❌ Expected error: {str(e)}")
        print("\n🚨 This demonstrates why Phase 1 is mandatory!")

def example_multiple_vector_stores():
    """Example of using multiple vector stores for different purposes"""
    print("\n🗂️ MULTIPLE VECTOR STORES EXAMPLE")
    print("=" * 60)
    
    # Create different vector stores for different purposes
    print("\n📦 Creating multiple vector stores...")
    
    # Vector store for general movies
    VectorStoreManager.create_vector_store(
        vector_store_path="./general_movies",
        collection_name="general_collection"
    )
    
    # Vector store for specific analysis
    VectorStoreManager.create_vector_store(
        vector_store_path="./analysis_movies",
        collection_name="analysis_collection"
    )
    
    # Use first vector store
    print("\n🎬 Using general movies vector store:")
    movie_finder_1 = EnhancedMovieFinder(
        vector_store_path="./general_movies",
        collection_name="general_collection"
    )
    movie_finder_1._initialize_system()
    response_1 = movie_finder_1.process_query("Find popular movies")
    print(f"Response 1: {response_1}")
    
    # Use second vector store
    print("\n🎬 Using analysis movies vector store:")
    movie_finder_2 = EnhancedMovieFinder(
        vector_store_path="./analysis_movies",
        collection_name="analysis_collection"
    )
    movie_finder_2._initialize_system()
    response_2 = movie_finder_2.process_query("Find critically acclaimed films")
    print(f"Response 2: {response_2}")

def example_load_existing_vector_store():
    """Example of loading an existing vector store"""
    print("\n🔄 LOADING EXISTING VECTOR STORE EXAMPLE")
    print("=" * 60)
    
    vector_store_path = "./existing_vectors"
    collection_name = "existing_movies"
    
    # First, create a vector store
    print("\n📦 Creating vector store...")
    VectorStoreManager.create_vector_store(
        vector_store_path=vector_store_path,
        collection_name=collection_name
    )
    
    # Now load it using VectorStoreManager
    print("\n🔄 Loading existing vector store...")
    try:
        vector_manager = VectorStoreManager(
            vector_store_path=vector_store_path,
            collection_name=collection_name
        )
        
        vector_store = vector_manager.load_existing_vector_store()
        print("✅ Vector store loaded successfully!")
        
        # Use with MovieFinder
        movie_finder = EnhancedMovieFinder(
            vector_store_path=vector_store_path,
            collection_name=collection_name
        )
        
        movie_finder._initialize_system()
        response = movie_finder.process_query("Find me movies about friendship")
        print(f"📝 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error loading vector store: {e}")

def main():
    """Main function demonstrating the two-phase approach"""
    print("🎬 MovieFinder System - Two-Phase Approach")
    print("=" * 60)
    
    print("""
📋 WORKFLOW OVERVIEW:
1. PHASE 1: Create vector store (mandatory first step)
2. PHASE 2: Process movie queries (uses existing vector store)

⚠️ IMPORTANT: You MUST complete Phase 1 before Phase 2!
""")
    
    try:
        # Run all examples
        phase_1_create_vector_store()
        phase_2_movie_processing()
        example_complete_workflow()
        example_error_handling()
        example_multiple_vector_stores()
        example_load_existing_vector_store()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 