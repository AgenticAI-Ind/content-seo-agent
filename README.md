# Content Creation & SEO Agent

A production-ready AI agent for automated content creation, SEO optimization, and multi-platform content repurposing using FastAPI, Ollama, and RAG (Retrieval-Augmented Generation).

## Features

- 🚀 **Multi-platform Content Generation** - Blog posts, LinkedIn, Twitter, Medium
- 🔍 **SEO Optimization** - Keyword research, meta descriptions, content scoring
- ♻️ **Content Repurposing** - Convert blog posts to threads, newsletters, social media
- 🧠 **RAG-Powered Research** - Uses vector embeddings for fact-based content
- 🎨 **Image Prompt Generation** - AI-generated image descriptions for content
- 📊 **Analytics Integration** - Track content performance metrics

## Tech Stack

- **FastAPI** - High-performance async API framework
- **Ollama** - Local LLM inference (llama3.2, mistral)
- **ChromaDB** - Vector database for RAG
- **LangChain** - LLM orchestration and RAG pipeline
- **BeautifulSoup4** - Web scraping for competitor analysis
- **Sentence Transformers** - Text embeddings

## Architecture

```
content-seo-agent/
├── src/
│   ├── agent/
│   │   ├── content_generator.py    # Core content generation logic
│   │   ├── seo_optimizer.py        # SEO analysis and optimization
│   │   ├── repurposer.py           # Content format conversion
│   │   └── rag_engine.py           # RAG implementation
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   └── routes.py               # API endpoints
│   ├── models/
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       ├── embeddings.py           # Vector embedding utilities
│       └── scraper.py              # Web scraping utilities
├── data/
│   └── knowledge_base/             # RAG knowledge base storage
├── tests/
│   └── test_agent.py
├── requirements.txt
├── .env.example
└── docker-compose.yml
```

## Installation

### Prerequisites

- Python 3.10+
- Ollama installed locally ([ollama.ai](https://ollama.ai))
- Docker (optional, for deployment)

### Setup

```bash
# Clone repository
cd content-seo-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama models
ollama pull llama3.2
ollama pull mistral

# Setup environment
cp .env.example .env
# Edit .env with your configuration

# Initialize knowledge base
python scripts/init_knowledge_base.py
```

## Usage

### Start the API Server

```bash
# Development mode with auto-reload
uvicorn src.api.main:app --reload --port 8000

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

#### Generate Blog Post
```bash
POST /api/v1/generate/blog
{
  "topic": "Introduction to AI Agents",
  "keywords": ["AI", "agents", "automation"],
  "tone": "professional",
  "length": 1500
}
```

#### SEO Analysis
```bash
POST /api/v1/seo/analyze
{
  "content": "Your article content here...",
  "target_keywords": ["AI agents", "automation"]
}
```

#### Repurpose Content
```bash
POST /api/v1/repurpose
{
  "source_content": "Blog post content...",
  "target_format": "twitter_thread",
  "max_tweets": 10
}
```

### Python SDK

```python
from content_agent import ContentAgent

# Initialize agent
agent = ContentAgent(model="llama3.2")

# Generate content with RAG
blog_post = agent.generate_blog(
    topic="AI in Healthcare",
    keywords=["AI", "healthcare", "diagnosis"],
    use_rag=True,
    tone="educational"
)

# SEO optimize
optimized = agent.optimize_seo(
    content=blog_post,
    target_keywords=["AI healthcare", "medical AI"]
)

# Repurpose to social media
thread = agent.repurpose(
    content=blog_post,
    format="twitter_thread"
)
```

## RAG Knowledge Base

The agent uses RAG to ground content in factual information:

1. **Add documents to knowledge base:**
```python
python scripts/add_to_kb.py --file research_paper.pdf
```

2. **Query knowledge base:**
```python
python scripts/query_kb.py --query "What are the benefits of AI agents?"
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Configuration

Edit `.env` file:

```env
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# ChromaDB Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8001

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Content Settings
DEFAULT_TONE=professional
MAX_CONTENT_LENGTH=3000
```

## Performance

- **Average generation time:** 15-30 seconds for 1000-word blog post
- **SEO analysis:** < 2 seconds
- **Repurposing:** 5-10 seconds
- **RAG retrieval:** < 1 second

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_agent.py::test_content_generation
```

## Roadmap

- [ ] Multi-language support (Spanish, French, German)
- [ ] Image generation integration (DALL-E, Stable Diffusion)
- [ ] WordPress/Medium auto-publishing
- [ ] A/B testing for content variations
- [ ] Content calendar scheduling
- [ ] Analytics dashboard

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Documentation: [docs.useagenticai.in](https://docs.useagenticai.in)
- Issues: [GitHub Issues](https://github.com/AgenticAI-Ind/content-seo-agent/issues)
- Email: support@useagenticai.in

---

Built with ❤️ by the AgenticAI team
