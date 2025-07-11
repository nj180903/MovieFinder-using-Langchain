# agents/segment_agent.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from typing import List, Optional
import traceback
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Define the expected structure of the filter config
class FilterSchema(BaseModel):
    """Schema for movie filter configuration"""
    genre: List[str] = Field(default=[], description="List of movie genres")
    actors: List[str] = Field(default=[], description="List of actor names")
    keywords: List[str] = Field(default=[], description="List of keywords to search in overview/plot")
    director: Optional[str] = Field(default=None, description="Director name")
    year_min: Optional[int] = Field(default=None, description="Minimum release year")
    year_max: Optional[int] = Field(default=None, description="Maximum release year")
    rating_min: Optional[float] = Field(default=None, description="Minimum IMDB rating")
    rating_max: Optional[float] = Field(default=None, description="Maximum IMDB rating")
    ranking_field: str = Field(default="IMDB_Rating", description="Field to sort by")
    sort_order: str = Field(default="desc", description="Sort order: asc or desc")
    limit: int = Field(default=10, description="Maximum number of results")

def get_segment_chain(llm):
    """Create segment chain that converts natural language to structured filters"""
    
    try:
        # Create a parser that checks the output against FilterSchema
        parser = PydanticOutputParser(pydantic_object=FilterSchema)

        # Enhanced prompt with better examples and instructions
        prompt = PromptTemplate(
            template="""
You are an expert at understanding movie queries and converting them into structured JSON filters.

Analyze the user query and extract relevant movie search criteria. Consider conversation history for context.

Query: "{query}"

Extract the following information:
- genre: List of movie genres mentioned (e.g., ["Action", "Thriller"])
- actors: List of actor names mentioned (e.g., ["Leonardo DiCaprio", "Brad Pitt"])
- keywords: List of keywords for plot/overview search (e.g., ["space", "alien", "war"])
- director: Director name if mentioned
- year_min, year_max: Year range if mentioned
- rating_min, rating_max: Rating range if mentioned
- ranking_field: What to sort by ("IMDB_Rating", "Released_Year", "Meta_score", "Gross")
- sort_order: "desc" for highest first, "asc" for lowest first
- limit: Number of results requested (default 10)
Examples:
- "top 5 action movies after 2010" → genre:["Action"], director:null, year_min:2010, ranking_field:"IMDB_Rating", limit:5
- "recent Leonardo DiCaprio thrillers" → actors:["Leonardo DiCaprio"], genre:["Thriller"], director:null, ranking_field:"Released_Year", limit:10
- "best rated sci-fi movies with space battles" → genre:["Sci-Fi"], keywords:["space", "battles"], director:null, ranking_field:"IMDB_Rating", limit:10

Output only valid JSON matching this format:
{format_instructions}
""",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Return a chain that runs the prompt + parses output
        return LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=parser
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating segment chain: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def create_fallback_filter(query: str) -> dict:
    """Create fallback filter when segment chain fails"""
    try:
        # Basic keyword-based fallback
        query_lower = query.lower()
        
        fallback = {
            "genre": [],
            "actors": [],
            "keywords": [],
            "director": "",
            "year_min": None,
            "year_max": None,
            "rating_min": None,
            "rating_max": None,
            "ranking_field": "IMDB_Rating",
            "sort_order": "desc",
            "limit": 10
        }
        
        # Extract genres
        genres = ["action", "comedy", "drama", "thriller", "horror", "sci-fi", "romance", "adventure"]
        for genre in genres:
            if genre in query_lower:
                fallback["genre"].append(genre.title())
        
        # Extract years
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if years:
            fallback["year_min"] = int(years[0])
        
        # Extract numbers for limits
        numbers = re.findall(r'\btop\s+(\d+)\b', query_lower)
        if numbers:
            fallback["limit"] = int(numbers[0])
        
        # Extract rating requirements
        if "best" in query_lower or "top" in query_lower:
            fallback["rating_min"] = 7.0
        
        return fallback
        
    except Exception as e:
        logger.error(f"❌ Error creating fallback filter: {str(e)}")
        return {
            "genre": [],
            "actors": [],
            "keywords": [],
            "director": "",
            "year_min": None,
            "year_max": None,
            "rating_min": None,
            "rating_max": None,
            "ranking_field": "IMDB_Rating",
            "sort_order": "desc",
            "limit": 10
        }