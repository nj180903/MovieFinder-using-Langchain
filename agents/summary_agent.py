# agents/summary_agent.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_summary_chain(llm):
    """Create summary chain for generating movie recommendations"""
    
    summary_prompt = PromptTemplate.from_template("""
You are a knowledgeable movie assistant. Based on the user's query, provide a helpful summary of the recommended movies.
Also guide user to give more detail or correct requirements if query is not clear.
You Always presence yourself in conversational manner if need greet or gratitude based on users conversation history.
User Query: {query}

Conversation history:
{context}
Found Movies:
{movies}

Instructions:
1. Start with a brief introduction addressing the user's specific request
2. Highlight the most relevant movies based on their query
3. Include key details like ratings, year, genre, and notable cast
4. Mention any patterns or themes in the recommendations
5. Keep the response engaging and informative
6. If fewer than expected results, explain why and suggest alternatives
7. Format the response in a conversational, helpful tone

Provide a comprehensive summary that helps the user understand why these movies match their request:
""")
    
    return summary_prompt | llm

def get_detailed_summary_chain(llm):
    """Create detailed summary chain for in-depth movie analysis"""
    
    detailed_prompt = PromptTemplate.from_template("""
You are a professional movie critic and recommendation expert. Provide a detailed analysis of the movies found.

User Query: {query}

Movies Found:
{movies}

Provide a comprehensive analysis including:
1. **Query Analysis**: What the user was looking for
2. **Top Recommendations**: Best matches with detailed explanations
3. **Notable Mentions**: Other interesting findings
4. **Patterns & Themes**: Common elements in the results
5. **Additional Suggestions**: Related movies they might enjoy

Format your response with clear sections and engaging descriptions that help the user discover their next favorite movie:
""")
    
    return detailed_prompt | llm

def format_movie_data(movies_data: list, max_movies: int = 5) -> str:
    """Format movie data for better presentation"""
    try:
        if not movies_data:
            return "No movie data available"
        
        formatted_movies = []
        for i, movie in enumerate(movies_data[:max_movies]):
            if isinstance(movie, str):
                # Clean up the movie text
                movie_text = movie.replace(" | ", "\n  ")
                formatted_movies.append(f"Movie {i+1}:\n  {movie_text}")
            else:
                formatted_movies.append(f"Movie {i+1}: {str(movie)}")
        
        return "\n\n".join(formatted_movies)
        
    except Exception as e:
        logger.error(f"❌ Error formatting movie data: {str(e)}")
        return "Error formatting movie data"