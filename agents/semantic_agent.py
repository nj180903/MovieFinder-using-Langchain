# agents/semantic_agent.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Dict, List, Tuple, Optional
from utils.logger import setup_logger
import re
import json

logger = setup_logger(__name__)

class SemanticQueryProcessor:
    """Enhanced semantic query processor with context awareness"""
    
    def __init__(self, llm):
        self.llm = llm
        self.conversation_context = []
        self.genre_synonyms = {
            'action': ['action', 'adventure', 'thriller', 'fast-paced', 'intense'],
            'comedy': ['comedy', 'funny', 'humor', 'hilarious', 'laugh', 'comic'],
            'drama': ['drama', 'emotional', 'serious', 'character-driven', 'deep'],
            'horror': ['horror', 'scary', 'frightening', 'terror', 'ghost', 'monster'],
            'romance': ['romance', 'love', 'romantic', 'relationship', 'couple'],
            'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien', 'robot'],
            'fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'dragon', 'mythical'],
            'crime': ['crime', 'criminal', 'detective', 'police', 'investigation'],
            'thriller': ['thriller', 'suspense', 'mystery', 'tension', 'psychological'],
            'war': ['war', 'military', 'battle', 'soldier', 'combat', 'conflict'],
            'western': ['western', 'cowboy', 'frontier', 'old west', 'gunfight'],
            'musical': ['musical', 'music', 'songs', 'dance', 'performance'],
            'animation': ['animation', 'animated', 'cartoon', 'family'],
            'documentary': ['documentary', 'real', 'factual', 'educational'],
            'biography': ['biography', 'biopic', 'life story', 'real person']
        }
    
    def process_query(self, query: str, conversation_history: List[Dict] = None) -> Dict:
        """Process query and return semantic search components"""
        try:
            logger.info(f"🔍 Processing semantic query: {query}")
            
            # Update conversation context
            if conversation_history:
                self.conversation_context = conversation_history[-5:]  # Keep last 5 exchanges
            
            # Generate semantic components
            semantic_sentence = self._generate_semantic_sentence(query)
            keywords = self._extract_enhanced_keywords(query)
            contextual_query = self._build_contextual_query(query, semantic_sentence, keywords)
            
            result = {
                'semantic_sentence': semantic_sentence,
                'keywords': keywords,
                'contextual_query': contextual_query,
                'search_strategy': self._determine_search_strategy(query)
            }
            
            logger.info(f"✅ Semantic processing completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in semantic processing: {str(e)}")
            return {
                'semantic_sentence': query,
                'keywords': self._extract_basic_keywords(query),
                'contextual_query': query,
                'search_strategy': 'basic'
            }
    
    def _generate_semantic_sentence(self, query: str) -> str:
        """Generate a semantic sentence optimized for vector search"""
        try:
            prompt = PromptTemplate.from_template("""
You are a movie search specialist. Transform the user's query into a semantic sentence that captures the essence of what they're looking for.

User Query: "{query}"
Conversation Context: {context}

Create a semantic sentence that:
1. Captures the main intent and themes
2. Includes relevant movie attributes (genre, mood, themes, setting)
3. Uses natural language that matches how movies are described
4. Incorporates context from previous conversation if relevant

Focus on creating a sentence that would match movie descriptions and plots effectively.

Examples:
- "funny movies about friendship" → "Comedy films featuring strong friendships, humor, and entertaining characters"
- "dark superhero movies" → "Dark and serious superhero films with psychological depth and intense themes"
- "romantic movies in Paris" → "Romantic films set in Paris featuring love stories and beautiful French settings"

Semantic Sentence:
""")
            
            context = self._get_context_summary()
            chain = prompt | self.llm
            
            result = chain.invoke({
                "query": query,
                "context": context
            })
            
            semantic_sentence = result.content if hasattr(result, 'content') else str(result)
            return semantic_sentence.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating semantic sentence: {str(e)}")
            return self._create_fallback_semantic_sentence(query)
    
    def _extract_enhanced_keywords(self, query: str) -> List[str]:
        """Extract enhanced keywords using NLP and domain knowledge"""
        try:
            keywords = set()
            query_lower = query.lower()
            
            # 1. Extract genre-based keywords
            for genre, synonyms in self.genre_synonyms.items():
                if any(synonym in query_lower for synonym in synonyms):
                    keywords.add(genre)
                    keywords.update(synonyms[:3])  # Add top 3 synonyms
            
            # 2. Extract mood and tone keywords
            mood_keywords = {
                'dark': ['dark', 'serious', 'gritty', 'noir', 'psychological'],
                'light': ['light', 'fun', 'entertaining', 'upbeat', 'cheerful'],
                'intense': ['intense', 'gripping', 'powerful', 'dramatic'],
                'funny': ['funny', 'hilarious', 'comic', 'humorous', 'witty'],
                'emotional': ['emotional', 'touching', 'heartwarming', 'moving'],
                'scary': ['scary', 'frightening', 'terrifying', 'chilling'],
                'exciting': ['exciting', 'thrilling', 'action-packed', 'adrenaline']
            }
            
            for mood, mood_terms in mood_keywords.items():
                if any(term in query_lower for term in mood_terms):
                    keywords.update(mood_terms[:2])
            
            # 3. Extract setting and time period keywords
            setting_patterns = {
                'space': ['space', 'alien', 'galaxy', 'planet', 'cosmos'],
                'medieval': ['medieval', 'knight', 'castle', 'kingdom', 'sword'],
                'modern': ['modern', 'contemporary', 'current', 'today'],
                'future': ['future', 'futuristic', 'sci-fi', 'technology'],
                'past': ['past', 'historical', 'period', 'vintage', 'classic'],
                'school': ['school', 'college', 'university', 'student', 'education'],
                'city': ['city', 'urban', 'metropolitan', 'downtown'],
                'war': ['war', 'battle', 'military', 'conflict', 'soldier']
            }
            
            for setting, setting_terms in setting_patterns.items():
                if any(term in query_lower for term in setting_terms):
                    keywords.update(setting_terms[:2])
            
            # 4. Extract character and relationship keywords
            character_patterns = {
                'family': ['family', 'father', 'mother', 'son', 'daughter', 'parent'],
                'friends': ['friend', 'friendship', 'buddy', 'companion'],
                'love': ['love', 'romance', 'couple', 'relationship'],
                'hero': ['hero', 'protagonist', 'main character'],
                'villain': ['villain', 'antagonist', 'bad guy'],
                'detective': ['detective', 'investigator', 'cop', 'police']
            }
            
            for char_type, char_terms in character_patterns.items():
                if any(term in query_lower for term in char_terms):
                    keywords.update(char_terms[:2])
            
            # 5. Extract specific movie-related terms
            movie_terms = ['movie', 'film', 'cinema', 'picture', 'flick']
            specific_words = [word for word in query_lower.split() 
                            if len(word) > 3 and word not in movie_terms]
            keywords.update(specific_words[:5])  # Add up to 5 specific terms
            
            # 6. Add contextual keywords from conversation history
            context_keywords = self._get_context_keywords()
            keywords.update(context_keywords)
            
            return list(keywords)
            
        except Exception as e:
            logger.error(f"❌ Error extracting enhanced keywords: {str(e)}")
            return self._extract_basic_keywords(query)
    
    def _build_contextual_query(self, original_query: str, semantic_sentence: str, 
                               keywords: List[str]) -> str:
        """Build a contextual query optimized for vector search"""
        try:
            # Get conversation context
            context_info = self._get_context_summary()
            
            # Create contextual query parts
            query_parts = []
            
            # 1. Add semantic sentence as primary content
            query_parts.append(semantic_sentence)
            
            # 2. Add high-priority keywords
            priority_keywords = [kw for kw in keywords if len(kw) > 3][:8]
            if priority_keywords:
                query_parts.append(f"Key themes: {', '.join(priority_keywords)}")
            
            # # 3. Add context if available
            # if context_info:
            #     query_parts.append(f"Context: {context_info}")
            
            # 4. Add original query essence
            # query_parts.append(f"User intent: {original_query}")
            
            contextual_query = " | ".join(query_parts)
            
            # Ensure query isn't too long (limit to ~500 chars for optimal performance)
            if len(contextual_query) > 500:
                contextual_query = contextual_query[:500] + "..."
            
            return contextual_query
            
        except Exception as e:
            logger.error(f"❌ Error building contextual query: {str(e)}")
            return semantic_sentence or original_query
    
    def _determine_search_strategy(self, query: str) -> str:
        """Determine the best search strategy for the query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['like', 'similar', 'same as', 'reminds me']):
            return 'similarity'
        elif any(word in query_lower for word in ['best', 'top', 'highest rated', 'recommend']):
            return 'recommendation'
        elif any(word in query_lower for word in ['genre', 'type', 'kind of', 'category']):
            return 'genre_based'
        elif any(word in query_lower for word in ['actor', 'director', 'cast', 'starring']):
            return 'person_based'
        elif any(word in query_lower for word in ['year', 'decade', 'recent', 'old', 'new']):
            return 'temporal'
        else:
            return 'general'
    
    def _get_context_summary(self) -> str:
        """Get a summary of conversation context"""
        if not self.conversation_context:
            return ""
        
        context_parts = []
        for message in self.conversation_context[-3:]:  # Last 3 messages
            if message.get('role') == 'user':
                context_parts.append(f"User asked: {message.get('content', '')}")
            elif message.get('role') == 'assistant':
                # Extract key themes from assistant response
                content = message.get('content', '')
                if 'genre' in content.lower():
                    context_parts.append("Previous discussion about genres")
                if any(word in content.lower() for word in ['recommend', 'suggest']):
                    context_parts.append("Previous recommendations given")
        
        return " | ".join(context_parts)
    
    def _get_context_keywords(self) -> List[str]:
        """Extract keywords from conversation context"""
        keywords = []
        
        for message in self.conversation_context:
            if message.get('role') == 'user':
                content = message.get('content', '').lower()
                # Extract genre mentions
                for genre in self.genre_synonyms.keys():
                    if genre in content:
                        keywords.append(genre)
                
                # Extract other relevant terms
                words = content.split()
                relevant_words = [word for word in words if len(word) > 4]
                keywords.extend(relevant_words[:3])  # Add up to 3 words per message
        
        return list(set(keywords))  # Remove duplicates
    
    def _create_fallback_semantic_sentence(self, query: str) -> str:
        """Create a fallback semantic sentence when LLM fails"""
        query_lower = query.lower()
        
        # Identify key components
        genres = []
        moods = []
        themes = []
        
        for genre, synonyms in self.genre_synonyms.items():
            if any(synonym in query_lower for synonym in synonyms):
                genres.append(genre)
        
        mood_indicators = {
            'funny': ['funny', 'comedy', 'humor'],
            'dark': ['dark', 'serious', 'gritty'],
            'exciting': ['exciting', 'thrilling', 'action'],
            'emotional': ['emotional', 'touching', 'heartwarming'],
            'scary': ['scary', 'horror', 'frightening']
        }
        
        for mood, indicators in mood_indicators.items():
            if any(ind in query_lower for ind in indicators):
                moods.append(mood)
        
        # Build fallback sentence
        parts = []
        if genres:
            parts.append(f"{' and '.join(genres)} movies")
        if moods:
            parts.append(f"with {' and '.join(moods)} themes")
        if not parts:
            parts.append("movies")
        
        return " ".join(parts)
    
    def _extract_basic_keywords(self, query: str) -> List[str]:
        """Extract basic keywords as fallback"""
        words = query.lower().split()
        return [word for word in words if len(word) > 3 and word not in ['movie', 'film', 'show']]

# Legacy functions for backward compatibility
def get_vector_phrase_chain(llm):
    """Legacy function - creates semantic search chain for extracting search keywords"""
    
    prompt = PromptTemplate.from_template("""
You are a semantic search assistant specialized in movie discovery.

Given the user query: "{query}"

Extract high-quality semantic keywords that can help match movie descriptions using text embeddings.

Focus on:
1. **Genres**: Action, Comedy, Drama, Thriller, Horror, Sci-Fi, Romance, etc.
2. **Themes**: Space, War, Love, Revenge, Friendship, Family, etc.
3. **Settings**: New York, Future, Medieval, Space, School, etc.
4. **Character Types**: Superhero, Detective, Spy, Alien, Robot, etc.
5. **Mood/Tone**: Dark, Funny, Emotional, Suspenseful, etc.
6. **Plot Elements**: Heist, Investigation, Journey, Battle, etc.

Return keywords as a comma-separated list, prioritizing the most relevant terms.
Avoid overly specific terms that might not match well.

Example:
Query: "funny movies about friendship in high school"
Keywords: comedy, friendship, high school, teenage, coming of age, funny, students, school

Keywords for your query:
""")
    
    return prompt | llm

def get_advanced_semantic_chain(llm):
    """Create advanced semantic chain with context awareness"""
    
    advanced_prompt = PromptTemplate.from_template("""
You are an advanced semantic search assistant with deep understanding of movie content.

User Query: "{query}"
Conversation Context: "{context}"

Extract semantic keywords optimized for vector similarity search:

1. **Primary Keywords** (most important): Core concepts that must be present
2. **Secondary Keywords** (supporting): Related concepts that enhance relevance
3. **Contextual Keywords** (from history): Keywords from previous conversation

Consider:
- Synonyms and related terms
- Genre conventions and tropes
- Cultural references and contexts
- Emotional undertones
- Visual and auditory elements

Format as three comma-separated lists:
Primary: [main keywords]
Secondary: [supporting keywords]
Contextual: [context keywords]
""")
    
    return LLMChain(llm=llm, prompt=advanced_prompt)

def extract_semantic_keywords(query: str) -> list:
    """Extract semantic keywords using rule-based approach as fallback"""
    try:
        # Common movie-related keywords
        genre_keywords = {
            'action': ['action', 'fight', 'battle', 'adventure', 'combat'],
            'comedy': ['funny', 'comedy', 'hilarious', 'humor', 'laugh'],
            'drama': ['drama', 'emotional', 'serious', 'life', 'family'],
            'thriller': ['thriller', 'suspense', 'mystery', 'tension'],
            'horror': ['horror', 'scary', 'fear', 'monster', 'ghost'],
            'romance': ['love', 'romance', 'relationship', 'romantic'],
            'sci-fi': ['sci-fi', 'science', 'future', 'space', 'alien', 'robot'],
            'fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'mythical']
        }
        
        keywords = []
        query_lower = query.lower()
        
        # Extract genre-based keywords
        for genre, terms in genre_keywords.items():
            if any(term in query_lower for term in terms):
                keywords.extend(terms[:3])  # Add top 3 related terms
        
        # Extract specific mentioned terms
        words = query_lower.split()
        movie_terms = ['movie', 'film', 'cinema', 'show', 'series']
        for word in words:
            if word not in movie_terms and len(word) > 3:
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
        
    except Exception as e:
        logger.error(f"❌ Error extracting semantic keywords: {str(e)}")
        return ['movie', 'film']