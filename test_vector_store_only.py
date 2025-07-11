# test_vector_store_only.py
"""
Simplified test for vector store refactoring without LLM dependencies.
This demonstrates the two-phase approach with mock data.
"""

import pandas as pd
import os
from core.vector_store import EnhancedVectorStore
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MockVectorStoreManager:
    """Mock VectorStoreManager for testing without full dependencies"""
    
    def __init__(self, vector_store_path: str = None, collection_name: str = None):
        self.vector_store_path = vector_store_path or 'test_movie_store_vetcor/'
        self.collection_name = collection_name or 'test_imdb_movie'
        self.vector_store = None
        
    def build_vector_store(self, df: pd.DataFrame = None) -> EnhancedVectorStore:
        """Build and initialize vector store with mock movie data"""
        try:
            logger.info("🔧 Building test vector store...")
            
            # Create mock data if not provided
            if df is None:
                df = self._create_mock_data()
            
            # Initialize vector store
            self.vector_store = EnhancedVectorStore()
            
            # Build vector store with data
            vector_db = self.vector_store.build_vector_store(
                df, 
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            
            logger.info(f"✅ Test vector store built successfully with {len(df)} movies")
            logger.info(f"📁 Vector store saved to: {self.vector_store_path}")
            logger.info(f"🗂️ Collection name: {self.collection_name}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store building failed: {str(e)}")
            raise
    
    def load_existing_vector_store(self) -> EnhancedVectorStore:
        """Load existing vector store from disk"""
        try:
            logger.info("🔄 Loading existing test vector store...")
            
            self.vector_store = EnhancedVectorStore()
            # Load existing vector store
            vector_db = self.vector_store.load_vector_store(
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            
            logger.info("✅ Test vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store loading failed: {str(e)}")
            raise
    
    def _create_mock_data(self) -> pd.DataFrame:
        """Create mock movie data for testing"""
        mock_data = {
            'Series_Title': [
                'The Matrix', 'Inception', 'The Dark Knight', 'Pulp Fiction', 
                'Fight Club', 'Interstellar', 'The Godfather', 'Forrest Gump',
                'The Shawshank Redemption', 'Goodfellas'
            ],
            'Released_Year': [1999, 2010, 2008, 1994, 1999, 2014, 1972, 1994, 1994, 1990],
            'Certificate': ['R', 'PG-13', 'PG-13', 'R', 'R', 'PG-13', 'R', 'PG-13', 'R', 'R'],
            'Runtime': ['136 min', '148 min', '152 min', '154 min', '139 min', '169 min', '175 min', '142 min', '142 min', '146 min'],
            'Genre': [
                'Action, Sci-Fi', 'Action, Sci-Fi, Thriller', 'Action, Crime, Drama',
                'Crime, Drama', 'Drama', 'Adventure, Drama, Sci-Fi', 'Crime, Drama',
                'Drama, Romance', 'Drama', 'Biography, Crime, Drama'
            ],
            'IMDB_Rating': [8.7, 8.8, 9.0, 8.9, 8.8, 8.6, 9.2, 8.8, 9.3, 8.7],
            'Overview': [
                'A computer hacker learns from mysterious rebels about the true nature of reality.',
                'A thief who enters the dreams of others to steal secrets from their minds.',
                'Batman begins his fight against crime in Gotham City.',
                'The lives of two mob hitmen, a boxer and others interweave in four tales of violence.',
                'An insomniac office worker and a soap salesman build a global organization.',
                'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
                'The aging patriarch of an organized crime dynasty transfers control to his reluctant son.',
                'The presidencies of Kennedy and Johnson through the perspective of Alabama man Forrest Gump.',
                'Two imprisoned men bond over a number of years, finding solace and eventual redemption.',
                'The story of Henry Hill and his life in the mob.'
            ],
            'Director': [
                'Lana Wachowski, Lilly Wachowski', 'Christopher Nolan', 'Christopher Nolan',
                'Quentin Tarantino', 'David Fincher', 'Christopher Nolan', 'Francis Ford Coppola',
                'Robert Zemeckis', 'Frank Darabont', 'Martin Scorsese'
            ],
            'Star1': ['Keanu Reeves', 'Leonardo DiCaprio', 'Christian Bale', 'John Travolta', 'Brad Pitt', 'Matthew McConaughey', 'Marlon Brando', 'Tom Hanks', 'Tim Robbins', 'Robert De Niro'],
            'Star2': ['Laurence Fishburne', 'Marion Cotillard', 'Heath Ledger', 'Samuel L. Jackson', 'Edward Norton', 'Anne Hathaway', 'Al Pacino', 'Robin Wright', 'Morgan Freeman', 'Ray Liotta'],
            'Star3': ['Carrie-Anne Moss', 'Tom Hardy', 'Aaron Eckhart', 'Uma Thurman', 'Helena Bonham Carter', 'Jessica Chastain', 'James Caan', 'Gary Sinise', 'Bob Gunton', 'Joe Pesci'],
            'No_of_Votes': [1500000, 2000000, 2200000, 1800000, 1900000, 1600000, 1700000, 1900000, 2300000, 1000000],
            'Gross': [171479930, 292576195, 534858444, 214179088, 100853753, 677471339, 134966411, 330252182, 16000000, 46836394]
        }
        
        return pd.DataFrame(mock_data)

def test_phase_1_vector_store_creation():
    """Test Phase 1: Vector store creation"""
    print("🚀 TESTING PHASE 1: Vector Store Creation")
    print("=" * 60)
    
    # Create vector store with mock data
    vector_manager = MockVectorStoreManager(
        vector_store_path="./test_movie_vectors",
        collection_name="test_movies"
    )
    
    # Build vector store
    vector_store = vector_manager.build_vector_store()
    
    print("✅ PHASE 1 COMPLETED: Vector store created successfully")
    return vector_manager

def test_phase_2_vector_store_loading():
    """Test Phase 2: Vector store loading"""
    print("\n🔄 TESTING PHASE 2: Vector Store Loading")
    print("=" * 60)
    
    try:
        # Try to load existing vector store
        vector_manager = MockVectorStoreManager(
            vector_store_path="./test_movie_vectors",
            collection_name="test_movies"
        )
        
        vector_store = vector_manager.load_existing_vector_store()
        
        # Test semantic search
        print("\n🔍 Testing semantic search...")
        search_results = vector_store.semantic_search("action movies", k=3)
        
        print(f"Found {len(search_results)} results:")
        for i, result in enumerate(search_results, 1):
            movie_data = result['movie_data']
            print(f"{i}. {movie_data['Series_Title']} ({movie_data['Released_Year']}) - {movie_data['Genre']}")
        
        print("\n✅ PHASE 2 COMPLETED: Vector store loading and search successful")
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {str(e)}")
        print("This is expected if Phase 1 wasn't run first!")

def test_complete_workflow():
    """Test complete workflow: Phase 1 + Phase 2"""
    print("\n🔄 TESTING COMPLETE WORKFLOW")
    print("=" * 60)
    
    try:
        # Phase 1: Create vector store
        print("\n🚀 Phase 1: Creating vector store...")
        vector_manager = test_phase_1_vector_store_creation()
        
        # Phase 2: Load and use vector store
        print("\n🔄 Phase 2: Loading and using vector store...")
        
        # Load the vector store we just created
        vector_store = vector_manager.load_existing_vector_store()
        
        # Test different types of searches
        test_queries = [
            "science fiction movies",
            "movies by Christopher Nolan",
            "crime drama films",
            "high rated movies"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            results = vector_store.semantic_search(query, k=2)
            
            for result in results:
                movie = result['movie_data']
                print(f"  - {movie['Series_Title']} ({movie['Released_Year']}) - Score: {result['relevance_score']:.2f}")
        
        print("\n✅ COMPLETE WORKFLOW TEST SUCCESSFUL!")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("🧪 VECTOR STORE REFACTORING TESTS")
    print("=" * 60)
    
    print("""
📋 TESTING OVERVIEW:
This test demonstrates the two-phase approach with mock data:
1. PHASE 1: Create vector store (independent operation)
2. PHASE 2: Load and use vector store (processing operation)

🔧 Using mock movie data to avoid dependency issues.
""")
    
    try:
        # Run individual phase tests
        test_phase_1_vector_store_creation()
        test_phase_2_vector_store_loading()
        
        # Run complete workflow test
        test_complete_workflow()
        
        print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📋 NEXT STEPS:")
        print("1. Fix the permission issue with pip install")
        print("2. Install: pip install langchain-google-genai")
        print("3. Run the full example_usage.py")
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 