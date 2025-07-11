# core/vector_store.py

import numpy as np
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from utils.logger import setup_logger
import pandas as pd
import re
import json
from data.imdb_loader import load_dataset
import traceback

logger = setup_logger(__name__)

class EnhancedVectorStore:
    """Enhanced vector store with intelligent storage and retrieval logic"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vector_store = None
        self.metadata_index = {}
        
    def _create_rich_content_representation(self, movie_record: Dict) -> str:
        """Create a rich, semantic content representation for optimal vector storage"""
        
        # Core movie information
        title = movie_record.get('Series_Title', '')
        overview = movie_record.get('Overview', '')
        genre = movie_record.get('Genre', '')
        director = movie_record.get('Director', '')
        
        # Cast information
        cast_members = []
        for i in range(1, 5):  # Star1 to Star4
            star = movie_record.get(f'Star{i}', '')
            if star and star.strip():
                cast_members.append(star)
        
        # Rating and year information
        rating = movie_record.get('IMDB_Rating', '')
        year = movie_record.get('Released_Year', '')
        certificate = movie_record.get('Certificate', '')
        
        # Create structured content with semantic richness
        content_parts = []
        
        # 1. Title and basic info
        content_parts.append(f"Movie: {title}")
        
        # 2. Genre and mood descriptors
        if genre:
            genres = [g.strip() for g in genre.split(',')]
            content_parts.append(f"Genres: {', '.join(genres)}")
            
            # Add semantic descriptors based on genre
            genre_descriptors = self._get_genre_descriptors(genres)
            if genre_descriptors:
                content_parts.append(f"Style: {', '.join(genre_descriptors)}")
        
        # 3. Plot and thematic content
        if overview:
            # Clean and enhance overview
            clean_overview = re.sub(r'\s+', ' ', overview.strip())
            content_parts.append(f"Plot: {clean_overview}")
            
            # Extract thematic keywords from overview
            themes = self._extract_themes(clean_overview)
            if themes:
                content_parts.append(f"Themes: {', '.join(themes)}")
        
        # 4. Cast and crew
        if director:
            content_parts.append(f"Director: {director}")
        
        if cast_members:
            content_parts.append(f"Cast: {', '.join(cast_members[:3])}")  # Top 3 stars
        
        # 5. Quality and era indicators
        if rating:
            try:
                rating_float = float(rating)
                if rating_float >= 8.0:
                    content_parts.append("Quality: Highly acclaimed, masterpiece")
                elif rating_float >= 7.0:
                    content_parts.append("Quality: Well-regarded, quality film")
                elif rating_float >= 6.0:
                    content_parts.append("Quality: Good entertainment value")
            except:
                pass
        
        if year:
            try:
                year_int = int(year)
                if year_int >= 2020:
                    content_parts.append("Era: Modern contemporary")
                elif year_int >= 2010:
                    content_parts.append("Era: Recent modern")
                elif year_int >= 2000:
                    content_parts.append("Era: Early 2000s")
                elif year_int >= 1990:
                    content_parts.append("Era: 1990s classic")
                elif year_int >= 1980:
                    content_parts.append("Era: 1980s retro")
                else:
                    content_parts.append("Era: Classic vintage")
            except:
                pass
        
        # 6. Audience and rating
        if certificate:
            content_parts.append(f"Rating: {certificate}")
        
        return " | ".join(content_parts)
    
    def _get_genre_descriptors(self, genres: List[str]) -> List[str]:
        """Get semantic descriptors based on genres"""
        descriptors = []
        genre_map = {
            'Action': ['fast-paced', 'thrilling', 'intense', 'adrenaline'],
            'Adventure': ['exciting', 'journey', 'exploration', 'quest'],
            'Comedy': ['funny', 'humorous', 'lighthearted', 'entertaining'],
            'Drama': ['emotional', 'serious', 'character-driven', 'deep'],
            'Horror': ['scary', 'frightening', 'suspenseful', 'dark'],
            'Thriller': ['tense', 'suspenseful', 'gripping', 'mysterious'],
            'Romance': ['romantic', 'love story', 'emotional', 'heartwarming'],
            'Sci-Fi': ['futuristic', 'technology', 'space', 'innovative'],
            'Fantasy': ['magical', 'mythical', 'imaginative', 'otherworldly'],
            'Crime': ['criminal', 'investigation', 'law enforcement', 'justice'],
            'Mystery': ['puzzling', 'investigative', 'enigmatic', 'detective'],
            'Western': ['frontier', 'cowboys', 'old west', 'rugged'],
            'War': ['military', 'conflict', 'battle', 'heroic'],
            'Musical': ['songs', 'music', 'performance', 'artistic'],
            'Animation': ['animated', 'family-friendly', 'creative', 'artistic'],
            'Documentary': ['factual', 'educational', 'real-life', 'informative'],
            'Family': ['family-friendly', 'wholesome', 'all-ages', 'heartwarming'],
            'Biography': ['real person', 'life story', 'historical', 'inspiring']
        }
        
        for genre in genres:
            if genre in genre_map:
                descriptors.extend(genre_map[genre][:2])  # Take top 2 descriptors
        
        return list(set(descriptors))  # Remove duplicates
    
    def _extract_themes(self, overview: str) -> List[str]:
        """Extract thematic keywords from movie overview"""
        themes = []
        
        # Common movie themes and keywords
        theme_patterns = {
            'friendship': ['friend', 'friendship', 'buddy', 'companion'],
            'love': ['love', 'romance', 'relationship', 'couple'],
            'family': ['family', 'father', 'mother', 'son', 'daughter', 'parent'],
            'revenge': ['revenge', 'vengeance', 'payback', 'retribution'],
            'justice': ['justice', 'law', 'police', 'detective', 'investigation'],
            'war': ['war', 'battle', 'military', 'soldier', 'combat'],
            'survival': ['survival', 'survive', 'escape', 'rescue'],
            'power': ['power', 'control', 'domination', 'authority'],
            'betrayal': ['betray', 'betrayal', 'deception', 'lie'],
            'redemption': ['redemption', 'forgiveness', 'second chance'],
            'coming-of-age': ['grow up', 'teenager', 'young', 'school'],
            'supernatural': ['ghost', 'spirit', 'supernatural', 'paranormal'],
            'technology': ['technology', 'computer', 'robot', 'artificial'],
            'space': ['space', 'alien', 'planet', 'galaxy', 'universe'],
            'time': ['time', 'past', 'future', 'history', 'timeline']
        }
        
        overview_lower = overview.lower()
        
        for theme, keywords in theme_patterns.items():
            if any(keyword in overview_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def build_vector_store(self, df: pd.DataFrame, persist_directory: str = None, 
                          collection_name: str = None) -> VectorStore:
        """Build vector store with enhanced content representation"""
        try:
            logger.info("🔧 Building enhanced vector store...")
            
            # Set default values if not provided
            if persist_directory is None:
                persist_directory = 'movie_store_vetcor/'
            if collection_name is None:
                collection_name = 'imdb_movie'
            
            documents = []
            self.metadata_index = {}
            
            for idx, row in df.iterrows():
                # Convert row to dictionary
                movie_record = row.to_dict()
                
                # Create rich content representation
                rich_content = self._create_rich_content_representation(movie_record)
                
                # Create metadata (excluding large text fields from metadata)
                metadata = {
                    'movie_id': idx,
                    'title': movie_record.get('Series_Title', ''),
                    'year': movie_record.get('Released_Year', ''),
                    'rating': movie_record.get('IMDB_Rating', ''),
                    'genre': movie_record.get('Genre', ''),
                    'director': movie_record.get('Director', ''),
                    'certificate': movie_record.get('Certificate', ''),
                    'runtime': movie_record.get('Runtime', ''),
                    'votes': movie_record.get('No_of_Votes', ''),
                    'gross': movie_record.get('Gross', '')
                }
                
                # Store full record for retrieval
                self.metadata_index[idx] = movie_record
                
                # Create document
                doc = Document(
                    page_content=rich_content,
                    metadata=metadata
                )
                documents.append(doc)
            
            # Build vector store
            self.vector_store = Chroma.from_documents(
                documents=documents, 
                embedding=self.embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            
            logger.info(f"✅ Vector store built successfully with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Error building vector store: {str(e)}")
            raise

    def load_vector_store(self, persist_directory: str = None, 
                         collection_name: str = None) -> VectorStore:
        """Load existing vector store from disk"""
        try:
            logger.info("🔄 Loading existing vector store...")
            
            # Set default values if not provided
            if persist_directory is None:
                persist_directory = 'movie_store_vetcor/'
            if collection_name is None:
                collection_name = 'imdb_movie'
            
            # Load existing vector store
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=persist_directory,
                collection_name=collection_name
            )
            
            # Try to get some documents to verify the store is loaded
            test_results = self.vector_store.similarity_search("test", k=1)
            
            if not test_results:
                logger.warning("⚠️ Vector store loaded but appears to be empty")
                
            # Reconstruct metadata_index from loaded documents
            self._reconstruct_metadata_index()
            
            logger.info("✅ Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Error loading vector store: {str(e)}")
            raise
    
    def _reconstruct_metadata_index(self):
        """Reconstruct metadata index from loaded vector store"""
        try:
            logger.info("🔄 Reconstructing metadata index...")
            
            # Get all documents from the vector store
            # We'll use a broad search to get all documents
            all_docs = self.vector_store.similarity_search("", k=10000)  # Get a large number
            
            self.metadata_index = {}
            
            for doc in all_docs:
                movie_id = doc.metadata.get('movie_id')
                if movie_id is not None:
                    # Reconstruct movie record from metadata
                    movie_record = {
                        'Series_Title': doc.metadata.get('title', ''),
                        'Released_Year': doc.metadata.get('year', ''),
                        'IMDB_Rating': doc.metadata.get('rating', ''),
                        'Genre': doc.metadata.get('genre', ''),
                        'Director': doc.metadata.get('director', ''),
                        'Certificate': doc.metadata.get('certificate', ''),
                        'Runtime': doc.metadata.get('runtime', ''),
                        'No_of_Votes': doc.metadata.get('votes', ''),
                        'Gross': doc.metadata.get('gross', ''),
                        # Add other fields as needed
                    }
                    
                    # Extract overview from content (this is a simplified approach)
                    # In a real implementation, you might want to store the overview separately
                    content = doc.page_content
                    if "Plot: " in content:
                        overview_start = content.find("Plot: ") + 6
                        overview_end = content.find(" | ", overview_start)
                        if overview_end == -1:
                            overview = content[overview_start:]
                        else:
                            overview = content[overview_start:overview_end]
                        movie_record['Overview'] = overview
                    
                    self.metadata_index[movie_id] = movie_record
            
            logger.info(f"✅ Metadata index reconstructed with {len(self.metadata_index)} entries")
            
        except Exception as e:
            logger.error(f"❌ Error reconstructing metadata index: {str(e)}")
            # Set empty index if reconstruction fails
            self.metadata_index = {}
    
    def semantic_search(self, query: str, k: int = 10, 
                       score_threshold: float = 0) -> List[Dict]:
        """Perform semantic search with enhanced ranking"""
        try:
            if not self.vector_store:
                raise ValueError("Vector store not initialized")
            
            logger.info(f"🔍 Performing semantic search for: {query}")
            
            # Get initial results with scores
            results = self.vector_store.similarity_search_with_score(
                query, k=k*2  # Get more results for re-ranking
            )
            logger.info(f"✅ Found {len(results)} relevant movies as semantic")
            # Filter by score threshold and re-rank
            filtered_results = []
            for doc, score in results:
                if score > score_threshold:  # Lower score = better match in most implementations
                    movie_data = self.metadata_index.get(doc.metadata['movie_id'])
                    if movie_data:
                        result = {
                            'movie_data': movie_data,
                            'metadata': doc.metadata,
                            'content': doc.page_content,
                            'similarity_score': score,
                            'relevance_score': self._calculate_relevance_score(query, doc, score)
                        }
                        filtered_results.append(result)
                        print(f"Result {json.dumps(result,indent = 3)}")
            
            # Sort by relevance score and return top k
            filtered_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            logger.info(f"✅ Found {len(filtered_results)} relevant movies")
            return filtered_results[:k]
            
        except Exception as e:
            logger.error(f"❌ Error in semantic search: {str(e)}")
            return []
    
    def _calculate_relevance_score(self, query: str, doc: Document, 
                                 similarity_score: float) -> float:
        """Calculate enhanced relevance score"""
        base_score = 1.0 - similarity_score  # Convert to 0-1 scale where 1 is best
        
        # Boost score based on rating
        try:
            rating = float(doc.metadata.get('rating', 0))
            rating_boost = min(rating / 10.0, 1.0)  # Normalize to 0-1
            base_score += rating_boost * 0.1  # 10% boost for rating
        except:
            pass
        
        # Boost score based on query-specific factors
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Boost for exact title matches
        title = doc.metadata.get('title', '').lower()
        if any(word in title for word in query_lower.split()):
            base_score += 0.2
        
        # Boost for genre matches
        genre = doc.metadata.get('genre', '').lower()
        if any(word in genre for word in query_lower.split()):
            base_score += 0.1
        
        # Boost for director matches
        director = doc.metadata.get('director', '').lower()
        if any(word in director for word in query_lower.split()):
            base_score += 0.15
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def hybrid_search(self, semantic_query: str, filters: Dict = None, 
                     k: int = 10) -> List[Dict]:
        """Combine semantic search with metadata filtering"""
        try:
            logger.info("🔄 Performing hybrid search...")
            
            # Get semantic results
            semantic_results = self.semantic_search(semantic_query, k=k*2)
            
            # Apply filters if provided
            if filters:
                filtered_results = []
                for result in semantic_results:
                    if self._matches_filters(result['metadata'], filters):
                        filtered_results.append(result)
                semantic_results = filtered_results
            
            logger.info(f"✅ Hybrid search completed: {len(semantic_results)} results")
            return semantic_results[:k]
            
        except Exception as e:
            logger.error(f"❌ Error in hybrid search: {str(e)}")
            return []
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check if metadata matches the given filters"""
        for key, value in filters.items():
            if key in metadata:
                meta_value = str(metadata[key]).lower()
                filter_value = str(value).lower()
                
                if isinstance(value, list):
                    # Check if any filter value matches
                    if not any(str(v).lower() in meta_value for v in value):
                        return False
                else:
                    # Check if filter value is in metadata
                    if filter_value not in meta_value:
                        return False
        return True

class VectorStoreManager:
    """Separate class for managing vector store initialization and building"""
    
    def __init__(self, vector_store_path: str = None, collection_name: str = None):
        self.vector_store_path = vector_store_path or 'movie_store_vetcor/'
        self.collection_name = collection_name or 'imdb_movie'
        self.vector_store = None
        self.df = None
        
    def build_vector_store(self, df: pd.DataFrame = None) -> EnhancedVectorStore:
        """Build and initialize vector store with movie data"""
        try:
            logger.info("🔧 Building vector store...")
            
            # Load dataset if not provided
            if df is None:
                logger.info("📊 Loading dataset for vector store...")
                self.df = load_dataset()
            else:
                self.df = df
            
            # Initialize vector store
            self.vector_store = EnhancedVectorStore()
            
            # Build vector store with data
            vector_db = self.vector_store.build_vector_store(
                self.df, 
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            
            logger.info(f"✅ Vector store built successfully with {len(self.df)} movies")
            logger.info(f"📁 Vector store saved to: {self.vector_store_path}")
            logger.info(f"🗂️ Collection name: {self.collection_name}")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store building failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_vector_store(self) -> EnhancedVectorStore:
        """Get existing vector store instance"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
        return self.vector_store
    
    def load_existing_vector_store(self) -> EnhancedVectorStore:
        """Load existing vector store from disk"""
        try:
            logger.info("🔄 Loading existing vector store...")
            
            self.vector_store = EnhancedVectorStore()
            # Load existing vector store
            vector_db = self.vector_store.load_vector_store(
                persist_directory=self.vector_store_path,
                collection_name=self.collection_name
            )
            
            logger.info("✅ Vector store loaded successfully")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"❌ Vector store loading failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    @staticmethod
    def create_vector_store(vector_store_path: str = None, collection_name: str = None, 
                           df: pd.DataFrame = None) -> None:
        """Static method to create vector store - Phase 1 operation"""
        logger.info("🚀 PHASE 1: Creating Vector Store")
        logger.info("=" * 50)
        
        manager = VectorStoreManager(vector_store_path, collection_name)
        manager.build_vector_store(df)
        
        logger.info("✅ PHASE 1 COMPLETED: Vector store created successfully")
        logger.info(f"📁 Path: {manager.vector_store_path}")
        logger.info(f"🗂️ Collection: {manager.collection_name}")
        logger.info("=" * 50)
