"""
SEO Optimizer for Content Agent
Analyzes and optimizes content for search engines.
"""

import re
from typing import List, Dict, Any
from collections import Counter
import textstat
from loguru import logger


class SEOOptimizer:
    """
    SEO analysis and optimization engine.

    Features:
    - Keyword density analysis
    - Readability scoring
    - Meta description generation
    - SEO recommendations
    """

    def __init__(self):
        """Initialize SEO optimizer"""
        logger.info("SEOOptimizer initialized")

    async def analyze(
        self,
        content: str,
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze content for SEO optimization.

        Args:
            content: Content to analyze
            target_keywords: List of target keywords

        Returns:
            Dictionary with SEO analysis results
        """
        logger.info(f"Analyzing content for SEO ({len(content)} chars)")

        # Calculate keyword density
        keyword_density = self._calculate_keyword_density(content, target_keywords)

        # Calculate readability score
        readability = self._calculate_readability(content)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            content,
            target_keywords,
            keyword_density
        )

        # Calculate overall SEO score
        score = self._calculate_seo_score(
            keyword_density,
            readability,
            content
        )

        # Generate meta description
        meta_description = self._generate_meta_description(content)

        return {
            "score": score,
            "keyword_density": keyword_density,
            "readability_score": readability,
            "recommendations": recommendations,
            "meta_description": meta_description
        }

    def _calculate_keyword_density(
        self,
        content: str,
        keywords: List[str]
    ) -> Dict[str, float]:
        """Calculate keyword density percentages"""
        content_lower = content.lower()
        words = re.findall(r'\w+', content_lower)
        total_words = len(words)

        densities = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            count = content_lower.count(keyword_lower)
            density = (count / total_words * 100) if total_words > 0 else 0
            densities[keyword] = round(density, 2)

        return densities

    def _calculate_readability(self, content: str) -> float:
        """Calculate Flesch Reading Ease score"""
        try:
            return round(textstat.flesch_reading_ease(content), 1)
        except:
            return 0.0

    def _generate_recommendations(
        self,
        content: str,
        keywords: List[str],
        density: Dict[str, float]
    ) -> List[str]:
        """Generate SEO improvement recommendations"""
        recommendations = []

        # Check keyword usage
        for keyword, dens in density.items():
            if dens < 0.5:
                recommendations.append(
                    f"Increase usage of '{keyword}' (current: {dens}%, target: 1-2%)"
                )
            elif dens > 3.0:
                recommendations.append(
                    f"Reduce usage of '{keyword}' to avoid keyword stuffing (current: {dens}%)"
                )

        # Check content length
        word_count = len(content.split())
        if word_count < 300:
            recommendations.append(
                f"Content is too short ({word_count} words). Aim for 800-1500 words."
            )

        # Check headings
        h2_count = content.count('## ')
        if h2_count < 2:
            recommendations.append("Add more H2 headings for better structure")

        # Check for images
        if '![' not in content:
            recommendations.append("Add images with descriptive alt text")

        return recommendations

    def _calculate_seo_score(
        self,
        keyword_density: Dict[str, float],
        readability: float,
        content: str
    ) -> int:
        """Calculate overall SEO score (0-100)"""
        score = 100

        # Deduct for poor keyword density
        for dens in keyword_density.values():
            if dens < 0.5 or dens > 3.0:
                score -= 10

        # Deduct for poor readability
        if readability < 50:
            score -= 15
        elif readability < 30:
            score -= 25

        # Deduct for short content
        word_count = len(content.split())
        if word_count < 300:
            score -= 20
        elif word_count < 500:
            score -= 10

        return max(0, score)

    def _generate_meta_description(self, content: str) -> str:
        """Generate SEO-optimized meta description"""
        # Extract first 2-3 sentences
        sentences = re.split(r'[.!?]+', content)
        description = ' '.join(sentences[:2]).strip()

        # Limit to 155 characters
        if len(description) > 155:
            description = description[:152] + '...'

        return description


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        optimizer = SEOOptimizer()

        sample_content = """
        # Introduction to AI Agents

        AI agents are autonomous software systems that can perceive their
        environment and take actions. These intelligent agents use machine
        learning and natural language processing.

        ## What are AI Agents?

        AI agents can perform tasks automatically without human intervention.
        They are used in various applications from chatbots to robotics.
        """

        result = await optimizer.analyze(
            content=sample_content,
            target_keywords=["AI agents", "machine learning"]
        )

        print(f"SEO Score: {result['score']}/100")
        print(f"Keyword Density: {result['keyword_density']}")
        print(f"Readability: {result['readability_score']}")
        print(f"Meta: {result['meta_description']}")

    asyncio.run(demo())
