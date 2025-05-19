from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.schema import Document

class AgentState(BaseModel):
    """Base state model for all agents."""
    documents: List[Document] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.state = AgentState()
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process the input data and return the result.
        
        Args:
            input_data: The input data to process
            
        Returns:
            The processed result
        """
        pass
    
    def update_state(self, **kwargs) -> None:
        """Update the agent's state with new information.
        
        Args:
            **kwargs: Key-value pairs to update in the state
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def add_error(self, error: str) -> None:
        """Add an error to the agent's state.
        
        Args:
            error: The error message to add
        """
        self.state.errors.append(error)
    
    def clear_errors(self) -> None:
        """Clear all errors from the agent's state."""
        self.state.errors = []
    
    def get_state(self) -> AgentState:
        """Get the current state of the agent.
        
        Returns:
            The current agent state
        """
        return self.state
    
    def reset_state(self) -> None:
        """Reset the agent's state to its initial values."""
        self.state = AgentState() 