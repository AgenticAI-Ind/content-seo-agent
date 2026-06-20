"""
FastAPI Application for Content & SEO Agent
Production-ready REST API with async support.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from loguru import logger
import sys

from ..agent.content_generator import ContentGenerator
from ..agent.seo_optimizer import SEOOptimizer
from ..agent.repurposer import ContentRepurposer
from ..models.schemas import (
    ContentRequest,
    ContentResponse,
    SEOAnalysisRequest,
    SEOAnalysisResponse,
    RepurposeRequest
)

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/api.log", rotation="500 MB", retention="10 days", level="DEBUG")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown.
    """
    # Startup
    logger.info("🚀 Starting Content & SEO Agent API")

    # Initialize agents
    app.state.content_generator = ContentGenerator(model_name="llama3.2")
    app.state.seo_optimizer = SEOOptimizer()
    app.state.repurposer = ContentRepurposer()

    logger.success("✅ All agents initialized successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Content & SEO Agent API")


# Initialize FastAPI app
app = FastAPI(
    title="Content & SEO Agent API",
    description="AI-powered content creation and SEO optimization",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Content & SEO Agent",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "agents": {
            "content_generator": "online",
            "seo_optimizer": "online",
            "repurposer": "online"
        }
    }


@app.post("/api/v1/generate/blog", response_model=ContentResponse)
async def generate_blog(request: ContentRequest):
    """
    Generate a complete blog post with SEO optimization.

    Args:
        request: ContentRequest with topic, keywords, tone, length

    Returns:
        ContentResponse with generated blog post
    """
    try:
        logger.info(f"Blog generation request: {request.topic}")

        generator = app.state.content_generator
        result = await generator.generate_blog_post(
            topic=request.topic,
            keywords=request.keywords,
            tone=request.tone,
            length=request.length,
            use_research=request.use_rag
        )

        return result

    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/social", response_model=ContentResponse)
async def generate_social(
    topic: str,
    platform: str = "linkedin",
    hashtags: list[str] = None
):
    """
    Generate platform-specific social media content.

    Args:
        topic: Content topic
        platform: Target platform (linkedin, twitter, instagram)
        hashtags: Optional hashtags

    Returns:
        ContentResponse with social media content
    """
    try:
        logger.info(f"Social post generation: {platform}")

        generator = app.state.content_generator
        result = await generator.generate_social_post(
            topic=topic,
            platform=platform,
            hashtags=hashtags or []
        )

        return result

    except Exception as e:
        logger.error(f"Social generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/seo/analyze", response_model=SEOAnalysisResponse)
async def analyze_seo(request: SEOAnalysisRequest):
    """
    Analyze content for SEO optimization opportunities.

    Args:
        request: SEOAnalysisRequest with content and target keywords

    Returns:
        SEOAnalysisResponse with SEO score and recommendations
    """
    try:
        logger.info("SEO analysis request")

        optimizer = app.state.seo_optimizer
        result = await optimizer.analyze(
            content=request.content,
            target_keywords=request.target_keywords
        )

        return result

    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/repurpose")
async def repurpose_content(request: RepurposeRequest):
    """
    Repurpose content to different formats.

    Args:
        request: RepurposeRequest with source content and target format

    Returns:
        Repurposed content in target format
    """
    try:
        logger.info(f"Repurpose request: {request.target_format}")

        repurposer = app.state.repurposer
        result = await repurposer.repurpose(
            source_content=request.source_content,
            target_format=request.target_format
        )

        return result

    except Exception as e:
        logger.error(f"Repurpose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
