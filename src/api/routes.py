"""
API Routes for Content & SEO Agent
Defines all API endpoints.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from loguru import logger

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

# Initialize routers
content_router = APIRouter(prefix="/api/v1/generate", tags=["Content Generation"])
seo_router = APIRouter(prefix="/api/v1/seo", tags=["SEO"])
repurpose_router = APIRouter(prefix="/api/v1/repurpose", tags=["Repurpose"])

# Initialize agents (these will be injected from main.py)
content_generator = None
seo_optimizer = None
repurposer = None


def init_routes(gen, seo, rep):
    """Initialize routes with agent instances"""
    global content_generator, seo_optimizer, repurposer
    content_generator = gen
    seo_optimizer = seo
    repurposer = rep


# Content Generation Routes

@content_router.post("/blog", response_model=ContentResponse)
async def generate_blog_post(request: ContentRequest):
    """
    Generate a complete blog post with SEO optimization.

    - **topic**: Main topic or title
    - **keywords**: List of target SEO keywords
    - **tone**: Writing tone (professional, casual, etc.)
    - **length**: Target word count
    - **use_rag**: Use RAG for research
    """
    try:
        logger.info(f"Blog generation request: {request.topic}")

        result = await content_generator.generate_blog_post(
            topic=request.topic,
            keywords=request.keywords,
            tone=request.tone.value,
            length=request.length,
            use_research=request.use_rag
        )

        return result

    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@content_router.post("/social", response_model=ContentResponse)
async def generate_social_post(
    topic: str,
    platform: str = "linkedin",
    hashtags: List[str] = None
):
    """
    Generate platform-specific social media content.

    - **topic**: Content topic
    - **platform**: Target platform (linkedin, twitter, instagram)
    - **hashtags**: Optional hashtags to include
    """
    try:
        logger.info(f"Social post generation: {platform}")

        result = await content_generator.generate_social_post(
            topic=topic,
            platform=platform,
            hashtags=hashtags or []
        )

        return result

    except Exception as e:
        logger.error(f"Social generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@content_router.post("/email", response_model=List[ContentResponse])
async def generate_email_sequence(
    campaign_goal: str,
    audience: str,
    num_emails: int = 5,
    tone: str = "friendly"
):
    """
    Generate a complete email marketing sequence.

    - **campaign_goal**: Goal of the email campaign
    - **audience**: Target audience description
    - **num_emails**: Number of emails in sequence
    - **tone**: Email tone
    """
    try:
        logger.info(f"Email sequence generation: {campaign_goal}")

        result = await content_generator.generate_email_sequence(
            campaign_goal=campaign_goal,
            audience=audience,
            num_emails=num_emails,
            tone=tone
        )

        return result

    except Exception as e:
        logger.error(f"Email generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# SEO Routes

@seo_router.post("/analyze", response_model=SEOAnalysisResponse)
async def analyze_seo(request: SEOAnalysisRequest):
    """
    Analyze content for SEO optimization opportunities.

    - **content**: Content to analyze
    - **target_keywords**: List of target keywords
    - **url**: Optional target URL
    """
    try:
        logger.info("SEO analysis request")

        result = await seo_optimizer.analyze(
            content=request.content,
            target_keywords=request.target_keywords
        )

        return result

    except Exception as e:
        logger.error(f"SEO analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@seo_router.get("/keywords/suggest")
async def suggest_keywords(
    topic: str,
    num_suggestions: int = 10
):
    """
    Get keyword suggestions for a topic.

    - **topic**: Main topic
    - **num_suggestions**: Number of keywords to suggest
    """
    try:
        # Placeholder for keyword research API
        return {
            "topic": topic,
            "suggestions": [
                f"{topic} guide",
                f"best {topic}",
                f"how to {topic}",
                f"{topic} tips",
                f"{topic} tutorial"
            ][:num_suggestions]
        }

    except Exception as e:
        logger.error(f"Keyword suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Repurpose Routes

@repurpose_router.post("")
async def repurpose_content(request: RepurposeRequest):
    """
    Repurpose content to different formats.

    - **source_content**: Original content
    - **target_format**: Target format (twitter_thread, linkedin_post, email, instagram_caption)
    - **max_length**: Optional maximum length
    """
    try:
        logger.info(f"Repurpose request: {request.target_format}")

        result = await repurposer.repurpose(
            source_content=request.source_content,
            target_format=request.target_format,
            max_length=request.max_length
        )

        return result

    except Exception as e:
        logger.error(f"Repurpose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check route

health_router = APIRouter(tags=["Health"])


@health_router.get("/health")
async def health_check():
    """
    Check API health status.
    """
    return {
        "status": "healthy",
        "service": "Content & SEO Agent",
        "version": "1.0.0",
        "agents": {
            "content_generator": "online" if content_generator else "offline",
            "seo_optimizer": "online" if seo_optimizer else "offline",
            "repurposer": "online" if repurposer else "offline"
        }
    }


# Export routers
all_routers = [content_router, seo_router, repurpose_router, health_router]
