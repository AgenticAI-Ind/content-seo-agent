"""
Content Generator Agent
Handles AI-powered content creation using Ollama and RAG.
"""

import asyncio
from typing import List, Optional, Dict, Any
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from loguru import logger

from .rag_engine import RAGEngine
from ..models.schemas import ContentRequest, ContentResponse


class ContentGenerator:
    """
    Main content generation agent with RAG capabilities.

    Features:
    - Multi-format content generation (blog, social, email)
    - RAG-powered research integration
    - Tone and style customization
    - SEO-optimized output
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        use_rag: bool = True
    ):
        """
        Initialize content generator.

        Args:
            model_name: Ollama model to use (llama3.2, mistral, etc.)
            temperature: Creativity level (0.0-1.0)
            use_rag: Enable RAG for fact-based generation
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_rag = use_rag

        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            num_ctx=4096  # Context window size
        )

        # Initialize RAG engine if enabled
        self.rag_engine = RAGEngine() if use_rag else None

        logger.info(f"ContentGenerator initialized with model: {model_name}")

    async def generate_blog_post(
        self,
        topic: str,
        keywords: List[str],
        tone: str = "professional",
        length: int = 1500,
        use_research: bool = True
    ) -> ContentResponse:
        """
        Generate a complete blog post with SEO optimization.

        Args:
            topic: Main topic/title of the blog post
            keywords: Target SEO keywords to include
            tone: Writing tone (professional, casual, technical, friendly)
            length: Target word count
            use_research: Use RAG to ground content in research

        Returns:
            ContentResponse with generated content and metadata
        """
        logger.info(f"Generating blog post on topic: {topic}")

        # Step 1: Retrieve relevant research if RAG is enabled
        research_context = ""
        sources = []

        if use_research and self.rag_engine:
            logger.debug("Retrieving relevant research from knowledge base")
            research_results = await self.rag_engine.retrieve(
                query=topic,
                top_k=5
            )
            research_context = "\n\n".join([
                f"Research {i+1}: {doc['text']}"
                for i, doc in enumerate(research_results)
            ])
            sources = [doc.get('source', 'Unknown') for doc in research_results]

        # Step 2: Create prompt template
        blog_prompt = PromptTemplate(
            input_variables=["topic", "keywords", "tone", "length", "research"],
            template="""You are a professional content writer and SEO expert.

Task: Write a comprehensive blog post on the following topic.

Topic: {topic}
Target Keywords: {keywords}
Tone: {tone}
Target Length: {length} words

{research}

Instructions:
1. Create an engaging title with the main keyword
2. Write a compelling introduction (hook the reader)
3. Organize content with clear H2 and H3 headings
4. Naturally incorporate all target keywords (avoid keyword stuffing)
5. Include actionable insights and examples
6. Write a strong conclusion with a call-to-action
7. Maintain the specified tone throughout
8. Aim for the target word count

Format the output in Markdown with proper headings, lists, and emphasis.

Blog Post:
"""
        )

        # Step 3: Generate content
        chain = LLMChain(llm=self.llm, prompt=blog_prompt)

        content = await asyncio.to_thread(
            chain.run,
            topic=topic,
            keywords=", ".join(keywords),
            tone=tone,
            length=length,
            research=f"\n\nRelevant Research:\n{research_context}" if research_context else ""
        )

        # Step 4: Extract metadata
        word_count = len(content.split())

        logger.success(f"Generated blog post: {word_count} words")

        return ContentResponse(
            content=content,
            word_count=word_count,
            metadata={
                "topic": topic,
                "keywords": keywords,
                "tone": tone,
                "sources": sources,
                "model": self.model_name
            }
        )

    async def generate_social_post(
        self,
        topic: str,
        platform: str = "linkedin",
        hashtags: Optional[List[str]] = None,
        include_hook: bool = True
    ) -> ContentResponse:
        """
        Generate platform-specific social media content.

        Args:
            topic: Topic or key message
            platform: Target platform (linkedin, twitter, instagram)
            hashtags: Optional hashtags to include
            include_hook: Add an attention-grabbing hook

        Returns:
            ContentResponse with platform-optimized content
        """
        logger.info(f"Generating {platform} post on: {topic}")

        # Platform-specific constraints
        platform_config = {
            "twitter": {"max_length": 280, "tone": "concise", "format": "thread"},
            "linkedin": {"max_length": 1300, "tone": "professional", "format": "post"},
            "instagram": {"max_length": 2200, "tone": "casual", "format": "caption"}
        }

        config = platform_config.get(platform, platform_config["linkedin"])

        social_prompt = PromptTemplate(
            input_variables=["topic", "platform", "max_length", "tone", "hook"],
            template="""You are a social media expert specializing in {platform}.

Task: Create an engaging {platform} post.

Topic: {topic}
Maximum Length: {max_length} characters
Tone: {tone}
{hook}

Instructions:
1. Start with a strong hook if requested
2. Make it scannable (use line breaks and emojis appropriately)
3. Include a clear call-to-action
4. Stay within the character limit
5. Optimize for {platform}'s algorithm and audience

{platform} Post:
"""
        )

        chain = LLMChain(llm=self.llm, prompt=social_prompt)

        content = await asyncio.to_thread(
            chain.run,
            topic=topic,
            platform=platform,
            max_length=config["max_length"],
            tone=config["tone"],
            hook="Include an attention-grabbing hook in the first line." if include_hook else ""
        )

        # Add hashtags if provided
        if hashtags:
            content += f"\n\n{' '.join(['#' + tag for tag in hashtags])}"

        return ContentResponse(
            content=content,
            word_count=len(content.split()),
            metadata={
                "platform": platform,
                "character_count": len(content),
                "hashtags": hashtags or []
            }
        )

    async def generate_email_sequence(
        self,
        campaign_goal: str,
        audience: str,
        num_emails: int = 5,
        tone: str = "friendly"
    ) -> List[ContentResponse]:
        """
        Generate a complete email marketing sequence.

        Args:
            campaign_goal: Goal of the email campaign
            audience: Target audience description
            num_emails: Number of emails in sequence
            tone: Email tone (friendly, professional, urgent)

        Returns:
            List of ContentResponse objects (one per email)
        """
        logger.info(f"Generating {num_emails}-email sequence for: {campaign_goal}")

        email_prompt = PromptTemplate(
            input_variables=["email_num", "total_emails", "goal", "audience", "tone"],
            template="""You are an email marketing expert.

Task: Write Email #{email_num} of a {total_emails}-email sequence.

Campaign Goal: {goal}
Target Audience: {audience}
Tone: {tone}

Instructions:
1. Create a compelling subject line (under 60 characters)
2. Start with a personalized greeting
3. Build on previous emails (if applicable)
4. Include a clear call-to-action
5. Keep it scannable with short paragraphs
6. End with a signature

Format:
Subject: [Your subject line]

[Email body]

Email:
"""
        )

        chain = LLMChain(llm=self.llm, prompt=email_prompt)

        # Generate emails sequentially
        emails = []
        for i in range(1, num_emails + 1):
            content = await asyncio.to_thread(
                chain.run,
                email_num=i,
                total_emails=num_emails,
                goal=campaign_goal,
                audience=audience,
                tone=tone
            )

            emails.append(ContentResponse(
                content=content,
                word_count=len(content.split()),
                metadata={
                    "email_number": i,
                    "total_emails": num_emails,
                    "campaign_goal": campaign_goal
                }
            ))

            logger.debug(f"Generated email {i}/{num_emails}")

        logger.success(f"Email sequence complete: {num_emails} emails")
        return emails

    async def generate_with_custom_prompt(
        self,
        custom_prompt: str,
        variables: Dict[str, Any]
    ) -> ContentResponse:
        """
        Generate content using a custom prompt template.

        Args:
            custom_prompt: Custom prompt with {variables}
            variables: Dictionary of variable values

        Returns:
            ContentResponse with generated content
        """
        logger.info("Generating with custom prompt")

        prompt = PromptTemplate(
            input_variables=list(variables.keys()),
            template=custom_prompt
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)
        content = await asyncio.to_thread(chain.run, **variables)

        return ContentResponse(
            content=content,
            word_count=len(content.split()),
            metadata={"custom_generation": True, "variables": variables}
        )


# Example usage
if __name__ == "__main__":
    async def demo():
        """Demo content generation"""
        generator = ContentGenerator(model_name="llama3.2")

        # Generate blog post
        blog = await generator.generate_blog_post(
            topic="The Future of AI Agents in Business",
            keywords=["AI agents", "business automation", "enterprise AI"],
            tone="professional",
            length=1500
        )

        print("=" * 80)
        print("BLOG POST GENERATED:")
        print("=" * 80)
        print(blog.content[:500] + "...")
        print(f"\nWord Count: {blog.word_count}")
        print(f"Sources: {blog.metadata.get('sources', [])}")

    asyncio.run(demo())
