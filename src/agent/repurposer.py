"""
Content Repurposer for Content Agent
Converts content between different formats.
"""

from typing import List, Dict, Any
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from loguru import logger


class ContentRepurposer:
    """
    Content format conversion engine.

    Features:
    - Blog to social media
    - Long-form to short-form
    - Text to email sequences
    - Format optimization
    """

    def __init__(self, model_name: str = "llama3.2"):
        """
        Initialize content repurposer.

        Args:
            model_name: Ollama model to use
        """
        self.model_name = model_name
        self.llm = Ollama(model=model_name, temperature=0.7)
        logger.info(f"ContentRepurposer initialized with {model_name}")

    async def repurpose(
        self,
        source_content: str,
        target_format: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Repurpose content to different format.

        Args:
            source_content: Original content
            target_format: Target format (twitter_thread, linkedin_post, email)
            **kwargs: Additional format-specific parameters

        Returns:
            Repurposed content with metadata
        """
        logger.info(f"Repurposing content to {target_format}")

        if target_format == "twitter_thread":
            return await self._to_twitter_thread(source_content, **kwargs)
        elif target_format == "linkedin_post":
            return await self._to_linkedin_post(source_content, **kwargs)
        elif target_format == "email":
            return await self._to_email(source_content, **kwargs)
        elif target_format == "instagram_caption":
            return await self._to_instagram(source_content, **kwargs)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")

    async def _to_twitter_thread(
        self,
        content: str,
        max_tweets: int = 10
    ) -> Dict[str, Any]:
        """Convert to Twitter thread"""
        prompt = PromptTemplate(
            input_variables=["content", "max_tweets"],
            template="""Convert the following content into a Twitter thread.

Rules:
- Each tweet must be under 280 characters
- Start with a hook tweet
- Number each tweet (1/n format)
- End with a CTA
- Maximum {max_tweets} tweets

Content:
{content}

Twitter Thread:
"""
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        thread = chain.run(content=content[:2000], max_tweets=max_tweets)

        # Parse into individual tweets
        tweets = []
        for line in thread.split('\n'):
            if line.strip():
                tweets.append(line.strip())

        return {
            "format": "twitter_thread",
            "content": thread,
            "tweets": tweets[:max_tweets],
            "tweet_count": len(tweets)
        }

    async def _to_linkedin_post(
        self,
        content: str,
        include_hashtags: bool = True
    ) -> Dict[str, Any]:
        """Convert to LinkedIn post"""
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Convert the following content into a LinkedIn post.

Rules:
- Professional tone
- Start with attention-grabbing first line
- Use line breaks for readability
- Maximum 1300 characters
- Add relevant hashtags at the end
- Include a clear CTA

Content:
{content}

LinkedIn Post:
"""
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        post = chain.run(content=content[:2000])

        return {
            "format": "linkedin_post",
            "content": post,
            "character_count": len(post)
        }

    async def _to_email(
        self,
        content: str,
        subject: str = None
    ) -> Dict[str, Any]:
        """Convert to email format"""
        prompt = PromptTemplate(
            input_variables=["content"],
            template="""Convert the following content into a professional email.

Rules:
- Create compelling subject line
- Professional greeting
- Clear and scannable body
- Strong CTA
- Professional signature placeholder

Content:
{content}

Email:
"""
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        email = chain.run(content=content[:2000])

        return {
            "format": "email",
            "content": email
        }

    async def _to_instagram(
        self,
        content: str,
        max_length: int = 2200
    ) -> Dict[str, Any]:
        """Convert to Instagram caption"""
        prompt = PromptTemplate(
            input_variables=["content", "max_length"],
            template="""Convert the following content into an Instagram caption.

Rules:
- Casual, engaging tone
- Start with emoji hook
- Use line breaks
- Add relevant emojis
- Maximum {max_length} characters
- Include hashtags

Content:
{content}

Instagram Caption:
"""
        )

        from langchain.chains import LLMChain
        chain = LLMChain(llm=self.llm, prompt=prompt)

        caption = chain.run(content=content[:1500], max_length=max_length)

        return {
            "format": "instagram_caption",
            "content": caption[:max_length],
            "character_count": len(caption)
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    async def demo():
        repurposer = ContentRepurposer()

        blog_content = """
        AI agents are transforming how we work. These autonomous systems
        can handle complex tasks, from customer service to data analysis.
        By leveraging machine learning, they continuously improve their
        performance and adapt to new situations.
        """

        # Convert to Twitter thread
        thread = await repurposer.repurpose(
            blog_content,
            "twitter_thread",
            max_tweets=5
        )

        print("Twitter Thread:")
        for i, tweet in enumerate(thread['tweets'], 1):
            print(f"{i}. {tweet}")

    asyncio.run(demo())
