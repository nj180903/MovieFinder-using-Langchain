# agents/filter_agent.py
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import traceback
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_filter_chain(llm):
    """Create filter chain that generates pandas code for DataFrame filtering"""
    
    prompt = PromptTemplate(
        template="""
You are a Python pandas expert. Given a JSON filter config, generate Python code that filters a DataFrame named `df`.

Filter Config:
{filters}

DataFrame Columns Available:
- Series_Title: Movie title
- Released_Year: Year of release
- Genre: Genres (may contain multiple genres like "Action, Thriller")
- Director: Director name
- Star1, Star2, Star3, Star4: Actor names
- IMDB_Rating: IMDb rating (float)
- Meta_score: Metacritic score (float)
- Overview: Movie description/plot
- Runtime: Movie runtime in minutes
- Gross: Box office earnings
- combined: Combined text of all movie info

IMPORTANT RULES:
1. Always assign the result to a variable named `filtered_df`
2. Use .query() method for string operations: df.query("Genre.str.contains('Action', na=False)")
3. Use .query() for numeric operations: df.query("Released_Year >= 2010")
4. For actor searches, check all Star columns: df.query("Star1.str.contains('Leo', na=False) | Star2.str.contains('Leo', na=False) | Star3.str.contains('Leo', na=False) | Star4.str.contains('Leo', na=False)")
5. Always use na=False in .str.contains() to handle NaN values
6. Sort by the ranking_field if provided
7. Apply limit using .head(n) at the end
8. Handle case-insensitive searches using case=False in .str.contains()

Example outputs:
- filtered_df = df.query("Genre.str.contains('Action', na=False, case=False)").head(5)
- filtered_df = df.query("Released_Year >= 2010 & IMDB_Rating >= 7.0").sort_values('IMDB_Rating', ascending=False).head(10)

Generate ONLY the pandas code, no explanations:
""",
        input_variables=["filters"]
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def generate_fallback_code(query: str) -> str:
    """Generate fallback pandas code when filter generation fails"""
    try:
        # Simple fallback based on common patterns
        if "top" in query.lower():
            return "filtered_df = df.sort_values('IMDB_Rating', ascending=False).head(10)"
        elif "recent" in query.lower() or "new" in query.lower():
            return "filtered_df = df.sort_values('Released_Year', ascending=False).head(10)"
        elif "rating" in query.lower():
            return "filtered_df = df.query('IMDB_Rating >= 7.0').sort_values('IMDB_Rating', ascending=False).head(10)"
        else:
            return "filtered_df = df.head(10)"
    except Exception as e:
        logger.error(f"❌ Error generating fallback code: {str(e)}")
        return "filtered_df = df.head(5)"