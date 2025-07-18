"""
Fallback handler for when the OpenAI API is not available.

This module provides a simple fallback mechanism that allows the application
to continue running with limited functionality when the OpenAI API is unavailable.
"""

from typing import Any, List, Mapping, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FallbackHandler(BaseChatModel):
    """
    A fallback handler that returns a predefined message when the OpenAI API is unavailable.
    This allows the application to continue running with limited functionality.
    """
    
    def _generate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs
    ) -> ChatResult:
        """Generate a response using the fallback handler."""
        fallback_message = """
        I'm currently operating in fallback mode due to an API authentication issue.
        
        To fix this issue:
        
        1. Check your OpenAI API key in the .env file
        2. Make sure your OpenAI account is active and has available credits
        3. Verify that the project you're trying to access hasn't been archived
        
        In the meantime, you can still use the data visualization tools with your uploaded dataset,
        but I won't be able to provide detailed analysis or respond to complex queries.
        
        If you're trying to visualize data, please specify exactly what type of visualization 
        you want (histogram, scatter plot, etc.) and which columns to use.
        """
        
        generation = ChatGeneration(
            message=BaseMessage(content=fallback_message, type="ai"),
            generation_info={"finish_reason": "fallback"}
        )
        return ChatResult(generations=[generation])
    
    async def _agenerate(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs
    ) -> ChatResult:
        """Asynchronously generate a response using the fallback handler."""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "fallback-handler"
