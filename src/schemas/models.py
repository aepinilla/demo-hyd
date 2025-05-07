"""
Data models and schemas for the chatbot application.

This module contains Pydantic models for data validation and serialization.
"""

from pydantic.v1 import BaseModel, Field
from typing import Optional, List, Dict, Any

class Message(BaseModel):
    """
    Model for a chat message.
    """
    role: str = Field(..., description="Role of the message sender (system, user, or assistant)")
    content: str = Field(..., description="Content of the message")

class ChatHistory(BaseModel):
    """
    Model for chat history.
    """
    messages: List[Message] = Field(default_factory=list, description="List of chat messages")

class FileRequest(BaseModel):
    """
    Model for a file reading request.
    """
    file_path: str = Field(..., description="Path to the file to read")

class ProcessMessageRequest(BaseModel):
    """
    Model for a message processing request.
    """
    message: str = Field(..., description="The message to process")
    context: Optional[str] = Field(None, description="Optional context for the message processing")
