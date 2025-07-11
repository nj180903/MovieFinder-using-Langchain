# 🎬 LangChain IMDb Multi-Agent Movie Finder

A movie recommendation system that uses LangChain, LLMs, and vector search to find movies based on natural language queries with conversation history and context awareness.

## 🚀 Features

- **Natural Language Queries**: Ask for movies using conversational language
- **Multi-Agent Architecture**: Specialized agents for different tasks
- **Conversation History**: Context-aware responses based on previous interactions
- **Vector Search**: Semantic similarity search for movie descriptions
- **Robust Error Handling**: Comprehensive error handling and logging
- **Real-time Processing**: Fast response times with efficient filtering
- **Web Interface**: Easy-to-use Gradio interface

## 🏗️ Architecture

The system consists of several specialized agents:

1. **Segment Agent**: Converts natural language to structured filters
2. **Filter Agent**: Generates pandas code for DataFrame operations
3. **Semantic Agent**: Extracts keywords for vector search
4. **Summary Agent**: Creates natural language summaries of results

## 📁 Project Structure

```
movie-finder/
├── main.py                          # Main application entry point
├── core/
│   ├── movie_finder.py             # Core MovieFinder class
│   └── vector_store.py             # Vector store operations
├── agents/
│   ├── segment_agent.py            # Query segmentation
│   ├── filter_agent.py             # Pandas code generation
│   ├── semantic_agent.py           # Semantic keyword extraction
│   └── summary_agent.py            # Result summarization
├── utils/
│   ├── config.py                   # Configuration management
│   ├── logger.py                   # Logging utilities
│   ├── conversation_history.py     # History management
│   ├── code_executor.py            # Safe code execution
│   └── error_handler.py            # Error handling utilities
├── data/
│   └── imdb_loader.py              # Dataset loading
├── logs/                           # Log files
├── requirements.txt                # Dependencies
├── .env.template                   # Environment variables template
└── README.md                       # This file
```

## 🛠️ Installation

1. **Dowload the Zip folder

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate 
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Edit .env with your GEMINI API KEY
```

5. **Prepare your dataset**  
   **Dataset Preparation**  
   This system uses the IMDb Top 1000 Movies and TV Shows Dataset from [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows).

   **Automatically Download with kagglehub**  
   We use the `kagglehub` package to download the dataset programmatically. It ensures the dataset is always available without manually uploading files.

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
GOOGLE_API_KEY=your_google_api_key_here
LOG_LEVEL=INFO
GRADIO_SERVER_PORT=7860
```

### Dataset Description

Dataset has following column:
- `Series_Title`: Movie title
- `Released_Year`: Year of release
- `Genre`: Movie genres
- `Director`: Director name
- `Star1`, `Star2`, `Star3`, `Star4`: Actor names
- `IMDB_Rating`: IMDb rating
- `Overview`: Movie description
- `combined`: Combined text of all movie information

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py 
```

The application will:
1. Initialize all components
2. Load the dataset
3. Set up the LLM and vector store
4. Launch the Gradio interface

### Example Queries

- "Top 5 action movies after 2010"
- "Leonardo DiCaprio thriller movies"
- "Best rated sci-fi movies about space"
- "Recent comedy movies with high ratings"
- "Movies similar to Inception"

### API Usage

```python
from core.movie_finder import MovieFinder

# Initialize the system
finder = MovieFinder()

# Process a query
result = finder.process_query("Top 5 action movies after 2010")
print(result)

# Get system information
info = finder.get_system_info()
print(info)
```

## 🔍 How It Works

1. **Query Processing**: User input is processed by the Segment Agent to extract structured filters
2. **Code Generation**: Filter Agent generates pandas code for DataFrame operations
3. **Safe Execution**: Code Executor safely runs the generated code with error handling
4. **Result Processing**: Results are processed and formatted for presentation
5. **Summary Generation**: Summary Agent creates natural language explanations
6. **History Management**: Conversation history is maintained for context

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **Safe Code Execution**: Validates and safely executes generated pandas code
- **Fallback Mechanisms**: Provides fallback options when primary methods fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring
- **User-Friendly Messages**: Clear error messages for users
- **Graceful Degradation**: System continues to work even with partial failures

## 📊 Logging

Logs are saved to `logs/movie_finder_YYYYMMDD.log` with different levels:
- `INFO`: General operation information
- `ERROR`: Error conditions with stack traces
- `DEBUG`: Detailed debugging information

## 🧪 Testing

Run tests with:
```bash
pytest tests/
```

## 🚀 Performance Optimization

- **Efficient Filtering**: Uses pandas query operations for fast DataFrame filtering
- **Vector Caching**: Caches vector embeddings for repeated queries
- **Memory Management**: Proper cleanup of resources
- **Async Operations**: Non-blocking operations where possible

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **API Key Issues**: Ensure your Google AI API key is correctly set in `.env`
2. **Dataset Loading**: Check that your dataset path and structure match the loader
3. **Memory Issues**: Reduce batch sizes or dataset size for large datasets
4. **Port Conflicts**: Change `GRADIO_SERVER_PORT` if 7860 is already in use

### Getting Help

- Check the logs in `logs/` directory
- Enable debug logging by setting `LOG_LEVEL=DEBUG`
- Review error messages in the Gradio interface
- Check system status using the status button

## 🔮 Future Enhancements

- [ ] Support for multiple datasets
- [ ] Advanced vector search with custom embeddings
- [ ] Integration with movie APIs for real-time data
- [ ] User preference learning
- [ ] Advanced recommendation algorithms
- [ ] Multi-language support
- [ ] Performance monitoring dashboard

## 📚 References

- [LangChain Documentation](https://docs.langchain.com/)
- [Gradio Documentation](https://gradio.app/docs/)
- [Google AI Documentation](https://ai.google.dev/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

# Enhanced Movie Finder - Vector Search Implementation

## Overview

This implementation provides a comprehensive movie recommendation system that combines semantic vector search with traditional filtering capabilities. The system intelligently chooses the best search strategy based on query characteristics.

## Key Features

### 🔧 Enhanced Vector Storage
- **Rich Content Representation**: Movies are stored with semantic descriptors, themes, and contextual information
- **Intelligent Metadata**: Structured metadata for hybrid search capabilities
- **Optimized Embeddings**: Uses sentence-transformers for high-quality vector representations

### 🧠 Semantic Query Processing
- **Context-Aware Processing**: Considers conversation history for better understanding
- **Multi-Strategy Approach**: Automatically selects optimal search strategy
- **Enhanced Keyword Extraction**: Domain-specific keyword extraction with synonyms

### 🎯 Search Strategies
1. **Semantic Search**: Vector similarity matching for conceptual queries
2. **Filter Search**: Pandas-based filtering for specific criteria
3. **Hybrid Search**: Combined approach for complex queries

## Architecture

```
User Query
    ↓
Query Analysis & Strategy Selection
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Semantic Search │ Filter Search   │ Hybrid Search   │
│                 │                 │                 │
│ • Vector DB     │ • Pandas Code   │ • Vector DB +   │
│ • Embedding     │ • Filters       │   Filters       │
│ • Similarity    │ • Structured    │ • Best of Both  │
└─────────────────┴─────────────────┴─────────────────┘
    ↓
Result Processing & Ranking
    ↓
Summary Generation