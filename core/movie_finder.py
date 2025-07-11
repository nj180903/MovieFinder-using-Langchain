# core/movie_finder.py
import traceback
import json
import re
import pandas as pd
from typing import List, Dict, Optional

from data.imdb_loader import load_dataset
from core.vector_store import EnhancedVectorStore
from utils.config import load_llm
from utils.logger import setup_logger
from utils.conversation_history import ConversationHistory
from utils.code_executor import CodeExecutor
from agents.segment_agent import get_segment_chain
from agents.filter_agent import get_filter_chain
from agents.semantic_agent import SemanticQueryProcessor
from agents.summary_agent import get_summary_chain
from utils.json_parser import _safe_json_extract
logger = setup_logger(__name__)

class EnhancedMovieFinder:
    """Enhanced movie finder system - PHASE 2: Processing only (no vector store creation)"""
    
    def __init__(self, vector_store_path: str, collection_name: str):
        """
        Initialize MovieFinder for processing phase.
        
        Args:
            vector_store_path: Path to existing vector store (required)
            collection_name: Name of existing collection (required)
        """
        if not vector_store_path:
            raise ValueError("vector_store_path is required. Vector store must be created first in Phase 1.")
        
        if not collection_name:
            raise ValueError("collection_name is required. Vector store must be created first in Phase 1.")
        
        self.history = ConversationHistory()
        self.code_executor = CodeExecutor()
        self.df = None
        self.llm = None
        self.semantic_processor = None
        self.chains = {}
        
        # Vector store configuration - only for loading existing stores
        self.vector_store_path = vector_store_path
        self.collection_name = collection_name
        self._vector_store = None
        
        logger.info("🎬 PHASE 2: MovieFinder initialized for processing")
        logger.info(f"📁 Vector store path: {self.vector_store_path}")
        logger.info(f"🗂️ Collection name: {self.collection_name}")
    
    def _initialize_system(self):
        """Initialize all system components (excluding vector store creation)"""
        try:
            logger.info("🚀 Initializing MovieFinder System for Processing...")
            
            # Load dataset
            logger.info("📊 Loading dataset...")
            self.df = load_dataset()
            logger.info(f"✅ Dataset loaded: {len(self.df)} movies")
            
            # Load LLM
            logger.info("🤖 Loading LLM...")
            self.llm = load_llm()
            logger.info("✅ LLM loaded successfully")
            
            # Initialize semantic processor
            logger.info("🧠 Initializing semantic processor...")
            self.semantic_processor = SemanticQueryProcessor(self.llm)
            logger.info("✅ Semantic processor initialized successfully")
            
            # Initialize chains
            logger.info("⛓️ Initializing agent chains...")
            self.chains = {
                'segment': get_segment_chain(self.llm),
                'filter': get_filter_chain(self.llm),
                'summary': get_summary_chain(self.llm)
            }
            logger.info("✅ All chains initialized successfully")
            
            # Load existing vector store
            logger.info("🔄 Loading existing vector store...")
            self._load_vector_store()
            logger.info("✅ Vector store loaded for processing")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_vector_store(self) -> None:
        """Load existing vector store - NO CREATION ALLOWED"""
        try:
            vector_store_obj = EnhancedVectorStore()
            # Load the vector store (this populates the internal Chroma object)
            vector_store_obj.load_vector_store(
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            # Store the wrapper object, not the raw Chroma object
            self._vector_store = vector_store_obj
            
        except Exception as e:
            error_msg = f"""
❌ Failed to load vector store from:
   Path: {self.vector_store_path}
   Collection: {self.collection_name}
   
🚨 IMPORTANT: You must create the vector store first using VectorStoreManager.create_vector_store()
   
Example:
   VectorStoreManager.create_vector_store(
       vector_store_path='{self.vector_store_path}',
       collection_name='{self.collection_name}'
   )
   
Error: {str(e)}
"""
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _get_vector_store(self) -> EnhancedVectorStore:
        """Get vector store instance - only loads existing, never creates"""
        if self._vector_store is None:
            raise ValueError("Vector store not loaded. Call _initialize_system() first.")
        
        # Debug: Check the actual type
        logger.info(f"🔍 Vector store type: {type(self._vector_store)}")
        
        # If somehow we got a raw Chroma object, wrap it
        if hasattr(self._vector_store, 'similarity_search') and not hasattr(self._vector_store, 'hybrid_search'):
            logger.warning("⚠️ Found raw Chroma object, wrapping it...")
            wrapper = EnhancedVectorStore()
            wrapper.vector_store = self._vector_store
            wrapper.metadata_index = {}
            wrapper._reconstruct_metadata_index()
            self._vector_store = wrapper
        
        return self._vector_store
    
    def _safe_json_extract(self, agent_output) -> dict:
        """Safely extract JSON from agent output"""
        try:
            content = agent_output.content if hasattr(agent_output, "content") else str(agent_output)
            # Remove markdown formatting
            clean_content = re.sub(r"```(?:json)?|```", "", content).strip()
            return json.loads(clean_content)
        except Exception as e:
            logger.error(f"❌ JSON extraction error: {str(e)}")
            logger.error(f"Raw content: {content}")
            return {}
    
    def _determine_search_strategy(self, query: str) -> str:
        """Determine the best search strategy based on query characteristics"""
        query_lower = query.lower()
        
        # Check for semantic indicators
        semantic_indicators = [
            'like', 'similar to', 'reminds me', 'mood', 'feel', 'atmosphere',
            'theme', 'about', 'featuring', 'with', 'emotional', 'funny', 'dark'
        ]
        
        # Check for filter indicators
        filter_indicators = [
            'year', 'rating', 'director', 'actor', 'genre', 'before', 'after',
            'rated', 'starring', 'by', 'from', 'between'
        ]
        
        semantic_score = sum(1 for indicator in semantic_indicators if indicator in query_lower)
        filter_score = sum(1 for indicator in filter_indicators if indicator in query_lower)
        
        if semantic_score > filter_score:
            return 'semantic'
        elif filter_score > semantic_score:
            return 'filter'
        else:
            return 'hybrid'
    
    def _semantic_search(self, query: str, context: str = "") -> str:
        """Perform semantic search using existing vector database"""
        try:
            logger.info("🔍 Performing semantic search...")
            
            # Get vector store instance
            vector_store = self._get_vector_store()
            
            # Process query with semantic processor
            semantic_result = self.semantic_processor.process_query(
                query, 
                self.history.get_recent_messages(5)
            )
            
            # Get search components
            contextual_query = semantic_result.get('contextual_query', query)
            search_strategy = semantic_result.get('search_strategy', 'general')
            
            logger.info(f"📝 Contextual query: {contextual_query}")
            logger.info(f"🎯 Search strategy: {search_strategy}")
            
            # Perform vector search
            search_results = vector_store.semantic_search(
                contextual_query,
                k=10,
                score_threshold=0.5
            )
            
            if not search_results:
                return "😕 No movies found matching your semantic criteria. Try rephrasing your query."
            
            # Prepare movie data for summary
            movies_data = []
            for result in search_results:
                movie_data = result['movie_data']
                movie_text = self._format_movie_for_summary(movie_data, result['relevance_score'])
                movies_data.append(movie_text)
            
            # Generate summary
            summary = self._generate_summary(query, movies_data)
            
            logger.info(f"✅ Semantic search completed: {len(search_results)} results")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Semantic search error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"❌ An error occurred during semantic search: {str(e)}"
    
    def _filter_search(self, query: str, context: str = "") -> str:
        """Perform filter-based search using pandas operations"""
        try:
            logger.info("🔧 Performing filter search...")
            
            # Generate pandas code
            pandas_code = self._generate_pandas_code(query, context)
            
            # Execute filter
            filtered_df = self.code_executor.execute_filter(self.df, pandas_code)
            
            if filtered_df.empty:
                return "😕 No movies found matching your filter criteria. Try adjusting your search terms."
            
            # Prepare movie data
            movies_data = self._prepare_movie_data(filtered_df)
            
            if not movies_data:
                return "😕 No movie data could be processed. Please try a different query."
            
            # Generate summary
            summary = self._generate_summary(query, movies_data)
            
            logger.info(f"✅ Filter search completed: {len(filtered_df)} results")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Filter search error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"❌ An error occurred during filter search: {str(e)}"
    
    def _hybrid_search(self, query: str, context: str = "") -> str:
        """Perform hybrid search combining semantic and filter approaches"""
        try:
            logger.info("🔄 Performing hybrid search...")
            
            # Get vector store instance
            vector_store = self._get_vector_store()
            
            # Process query with semantic processor
            semantic_result = self.semantic_processor.process_query(
                query, 
                self.history.get_recent_messages(5)
            )
            
            # Extract potential filters from query
            segment_result = self.chains['segment'].invoke({
                "query": f"{context}\nCurrent query: {query}"
            })
            
            filter_config = self._safe_json_extract(segment_result)
            
            # Determine if we have meaningful filters
            has_filters = any([
                filter_config.get('genre'),
                filter_config.get('actors'),
                filter_config.get('director'),
                filter_config.get('year_min'),
                filter_config.get('year_max'),
                filter_config.get('rating_min'),
                filter_config.get('rating_max'),
                filter_config.get('keywords')
            ])
            
            if has_filters:
                # Use hybrid search from vector store
                contextual_query = semantic_result.get('contextual_query', query)
                search_results = vector_store.hybrid_search(
                    contextual_query,
                    filters=filter_config,
                    k=10
                )
                
                if search_results:
                    movies_data = []
                    for result in search_results:
                        movie_data = result['movie_data']
                        movie_text = self._format_movie_for_summary(movie_data, result['relevance_score'])
                        movies_data.append(movie_text)
                    
                    summary = self._generate_summary(query, movies_data)
                    logger.info(f"✅ Hybrid search completed: {len(search_results)} results")
                    return summary
                else:
                    # Fallback to semantic search
                    return self._semantic_search(query, context)
            else:
                # No meaningful filters, use semantic search
                return self._semantic_search(query, context)
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"❌ An error occurred during hybrid search: {str(e)}"
    
    def _format_movie_for_summary(self, movie_data: Dict, relevance_score: float) -> str:
        """Format movie data for summary generation"""
        try:
            parts = []
            
            # Title and basic info
            title = movie_data.get('Series_Title', 'Unknown')
            year = movie_data.get('Released_Year', 'Unknown')
            rating = movie_data.get('IMDB_Rating', 'N/A')
            
            parts.append(f"Title: {title} ({year}) - Rating: {rating}")
            
            # Genre and director
            genre = movie_data.get('Genre', 'Unknown')
            director = movie_data.get('Director', 'Unknown')
            parts.append(f"Genre: {genre} | Director: {director}")
            
            # Cast
            cast = []
            for i in range(1, 4):  # Star1 to Star3
                star = movie_data.get(f'Star{i}', '')
                if star and star.strip():
                    cast.append(star)
            
            if cast:
                parts.append(f"Cast: {', '.join(cast)}")
            
            # Overview
            overview = movie_data.get('Overview', '')
            if overview:
                # Truncate overview if too long
                if len(overview) > 200:
                    overview = overview[:200] + "..."
                parts.append(f"Plot: {overview}")
            
            # Relevance score (for debugging)
            parts.append(f"Relevance: {relevance_score:.2f}")
            
            return " | ".join(parts)
            
        except Exception as e:
            logger.error(f"❌ Error formatting movie data: {str(e)}")
            return f"Movie: {movie_data.get('Series_Title', 'Unknown')}"
    
    def _generate_pandas_code(self, query: str, context: str = "") -> str:
        """Generate pandas code using the filter agent"""
        try:
            logger.info("🔧 Generating pandas filter code...")
            
            # First, get structured filters from segment agent
            segment_result = self.chains['segment'].invoke({
                "query": f"{context}\nCurrent query: {query}"
            })
            
            filter_config = self._safe_json_extract(segment_result)
            logger.info(f"📋 Filter config: {filter_config}")
            
            # Generate pandas code using filter agent
            filter_result = self.chains['filter'].invoke({
                "filters": json.dumps(filter_config, indent=2)
            })
            
            pandas_code = filter_result.content if hasattr(filter_result, "content") else str(filter_result)
            
            # Clean up the code
            pandas_code = re.sub(r"```python|```", "", pandas_code).strip()
            
            logger.info("✅ Pandas code generated successfully")
            return pandas_code
            
        except Exception as e:
            logger.error(f"❌ Pandas code generation error: {str(e)}")
            logger.error(traceback.format_exc())
            # Return a basic fallback query
            return "filtered_df = df.head(10)"
    
    def _generate_summary(self, query: str, movies_data: List[str]) -> str:
        """Generate summary using summary agent"""
        try:
            logger.info("📝 Generating summary...")
            context = self.history.get_context()
            movies_text = "\n\n".join(movies_data[:5])  # Limit to avoid token limits
            summary_result = self.chains['summary'].invoke({
                "query": query,
                "movies": movies_text,
                "context":context
            })
            
            summary = summary_result.content if hasattr(summary_result, "content") else str(summary_result)
            logger.info(f"✅ Summary generated successfully\n{summary}")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Summary generation error: {str(e)}")
            logger.error(traceback.format_exc())
            return f"Found {len(movies_data)} movies matching your query."
    
    def _prepare_movie_data(self, filtered_df: pd.DataFrame) -> List[str]:
        """Prepare movie data for summary generation"""
        try:
            movies_data = []
            
            if 'combined' in filtered_df.columns:
                movies_data = filtered_df['combined'].head(10).tolist()
            else:
                # Fallback: create combined text from available columns
                text_columns = ['Series_Title', 'Overview', 'Genre', 'Director', 'Star1', 'Star2',"Release_year",""]
                available_columns = [col for col in text_columns if col in filtered_df.columns]
                
                for _, row in filtered_df.head(10).iterrows():
                    movie_text = " | ".join([
                        f"{col}: {row[col]}" 
                        for col in available_columns 
                        if pd.notna(row[col])
                    ])
                    movies_data.append(movie_text)
            
            return movies_data
            
        except Exception as e:
            logger.error(f"❌ Error preparing movie data: {str(e)}")
            logger.error(traceback.format_exc())
            return []
    
    def process_query(self, query: str) -> str:
        """Main query processing pipeline - PROCESSING ONLY"""
        try:
            logger.info(f"🎬 Processing query: {query}")
            
            # Add user query to history
            self.history.add_message("user", query)
            
            # Get conversation context
            context = self.history.get_context()
            
            # Determine search strategy
            search_strategy = self._determine_search_strategy(query)
            logger.info(f"🎯 Selected search strategy: {search_strategy}")
            
            # Execute appropriate search strategy
            if search_strategy == 'semantic':
                response = self._semantic_search(query, context)
            elif search_strategy == 'filter':
                response = self._filter_search(query, context)
            else:  # hybrid
                response = self._hybrid_search(query, context)
            
            # Add response to history
            self.history.add_message("assistant", response)
            
            logger.info(f"✅ Query processed successfully using {search_strategy} strategy")
            return response
            
        except Exception as e:
            error_msg = f"❌ An error occurred while processing your query: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.history.add_message("assistant", error_msg)
            return error_msg
    
    def get_system_info(self) -> str:
        """Get system information for debugging"""
        try:
            vector_store_status = "❌ Not loaded"
            try:
                vector_store = self._get_vector_store()
                vector_store_status = "✅ Ready"
            except Exception:
                pass
            
            info = f"""
🎬 Enhanced Movie Finder System Status (PHASE 2 - Processing):
📊 Dataset: {len(self.df) if self.df is not None else 0} movies loaded
🤖 LLM: {'✅ Ready' if self.llm else '❌ Not loaded'}
🔧 Vector Store: {vector_store_status}
🗂️ Vector Store Path: {self.vector_store_path}
📁 Collection Name: {self.collection_name}
🧠 Semantic Processor: {'✅ Ready' if self.semantic_processor else '❌ Not loaded'}
⛓️ Chains: {len(self.chains)} agents initialized
💬 History: {len(self.history.messages)} messages
📊 Available columns: {', '.join(self.df.columns.tolist()) if self.df is not None else 'None'}

🎯 Search Strategies Available:
- Semantic Search: Vector similarity matching
- Filter Search: Pandas-based filtering
- Hybrid Search: Combined semantic + filter approach

⚠️ IMPORTANT: This is Phase 2 - Processing only. Vector store must be created first.
            """
            return info
        except Exception as e:
            return f"❌ Error getting system info: {str(e)}"
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        logger.info("✅ Conversation history cleared")

# For backward compatibility, keep the original MovieFinder class
class MovieFinder(EnhancedMovieFinder):
    """Legacy MovieFinder class for backward compatibility"""
    pass