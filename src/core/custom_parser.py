"""
Custom parser and output formatter for the ReAct agent.

This module provides a custom output parser that handles formatting errors
more gracefully for better user experience.
"""

from typing import Union, List, Dict, Any, Optional
from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

class GracefulReActOutputParser(AgentOutputParser):
    """
    A more forgiving output parser for ReAct agents that handles common formatting errors.
    """
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """
        Parse the output text into an AgentAction or AgentFinish.
        
        Args:
            text: The agent's output text
            
        Returns:
            Union[AgentAction, AgentFinish]: Either an action to take or the final answer
        """
        # Check if the output contains a final answer
        if "Final Answer:" in text:
            # Extract everything after "Final Answer:"
            answer_text = text.split("Final Answer:")[-1].strip()
            return AgentFinish(
                return_values={"output": answer_text},
                log=text
            )
            
        # Check for the Action/Action Input pattern
        if "Action:" in text and "Action Input:" in text:
            # Split by Action: and get the second part
            action_part = text.split("Action:")[-1].strip()
            
            # Extract the action name (everything before "Action Input:")
            action_name = action_part.split("Action Input:")[0].strip()
            
            # Extract the action input (everything after "Action Input:")
            action_input_str = action_part.split("Action Input:")[-1].strip()
            
            # Return the action
            return AgentAction(
                tool=action_name,
                tool_input=action_input_str,
                log=text
            )
        
        # Handle common formatting errors
        
        # Case 1: Has Thought but missing Action/Final Answer
        if "Thought:" in text and not any(marker in text for marker in ["Action:", "Final Answer:"]):
            # Treat it as a final answer
            return AgentFinish(
                return_values={"output": text.strip()},
                log=text
            )
            
        # Case 2: Has direct tool call without proper formatting
        for tool_marker in ["plot_", "analyze_", "fetch_", "compare_", "create_"]:
            if f"{tool_marker}" in text and ":" in text and "{" in text:
                # Try to extract tool name and input
                parts = text.split(":", 1)
                potential_tool = parts[0].strip()
                potential_input = parts[1].strip()
                
                if potential_tool and potential_input:
                    return AgentAction(
                        tool=potential_tool,
                        tool_input=potential_input,
                        log=text
                    )
        
        # Default to treating the whole text as a final answer
        return AgentFinish(
            return_values={"output": text.strip()},
            log=text
        )
