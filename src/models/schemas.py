"""
Pydantic schemas for Content & SEO Agent
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ToneType(str, Enum):
    """Content tone options"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    TECHNICAL = "technical"
    EDUCATIONAL = "educational"
    PERSUASIVE = "persuasive"


class PlatformType(str, Enum):
    """Social media platforms"""
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


class ContentRequest(BaseModel):
    """Request schema for content generation"""
    topic: str = Field(..., description="Main topic or title", min_length=3, max_length=200)
    keywords: List[str] = Field(..., description="Target SEO keywords", min_items=1, max_items=10)
    tone: ToneType = Field(default=ToneType.PROFESSIONAL, description="Writing tone")
    length: int = Field(default=1500, description="Target word count", ge=100, le=5000)
    use_rag: bool = Field(default=True, description="Use RAG for research")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Introduction to AI Agents",
                "keywords": ["AI", "agents", "automation", "business"],
                "tone": "professional",
                "length": 1500,
                "use_rag": True
            }
        }


class ContentResponse(BaseModel):
    """Response schema for generated content"""
    content: str = Field(..., description="Generated content")
    word_count: int = Field(..., description="Actual word count")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "# Introduction to AI Agents\n\nAI agents are...",
                "word_count": 1523,
                "metadata": {
                    "topic": "AI Agents",
                    "keywords": ["AI", "agents"],
                    "sources": ["source1.pdf", "source2.md"]
                }
            }
        }


class SEOAnalysisRequest(BaseModel):
    """Request schema for SEO analysis"""
    content: str = Field(..., description="Content to analyze", min_length=100)
    target_keywords: List[str] = Field(..., description="Target keywords to check")
    url: Optional[str] = Field(None, description="Optional target URL")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Long article content here...",
                "target_keywords": ["AI agents", "automation"],
                "url": "https://example.com/article"
            }
        }


class SEOAnalysisResponse(BaseModel):
    """Response schema for SEO analysis"""
    score: int = Field(..., description="SEO score (0-100)", ge=0, le=100)
    keyword_density: Dict[str, float] = Field(..., description="Keyword density percentages")
    recommendations: List[str] = Field(..., description="SEO improvement recommendations")
    readability_score: float = Field(..., description="Readability score")
    meta_description: Optional[str] = Field(None, description="Suggested meta description")

    class Config:
        json_schema_extra = {
            "example": {
                "score": 85,
                "keyword_density": {
                    "AI agents": 2.3,
                    "automation": 1.8
                },
                "recommendations": [
                    "Add more internal links",
                    "Increase keyword in H2 headings"
                ],
                "readability_score": 65.2,
                "meta_description": "Learn about AI agents and automation..."
            }
        }


class RepurposeRequest(BaseModel):
    """Request schema for content repurposing"""
    source_content: str = Field(..., description="Source content to repurpose", min_length=100)
    target_format: str = Field(..., description="Target format (twitter_thread, linkedin_post, email)")
    max_length: Optional[int] = Field(None, description="Maximum length for target format")

    class Config:
        json_schema_extra = {
            "example": {
                "source_content": "Blog post content here...",
                "target_format": "twitter_thread",
                "max_length": 280
            }
        }
