"""
Simplified schemas for the basic chatbot application.
"""

from pydantic.v1 import BaseModel, Field
from typing import Optional

class ReadFile(BaseModel):
    """
    Pydantic arguments schema for read_file method
    """
    file_path: str = Field(..., description="Path to the file to read.")

class ProcessMessage(BaseModel):
    """
    Pydantic arguments schema for process_message method
    """
    message: str = Field(..., description="The message to process.")
    context: Optional[str] = Field(None, description="Optional context for the message processing.")
