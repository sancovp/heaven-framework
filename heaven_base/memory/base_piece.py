from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, model_validator

class BasePiece(BaseModel):
    """Base class for all pieces with common functionality"""
    created_datetime: datetime = Field(default_factory=datetime.now)
    metadata: str = ""
    project: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_project(self) -> 'BasePiece':
        """Ensure project is set for non-Project pieces"""
        cls_name = self.__class__.__name__
        if cls_name != "Project" and not self.project:
            self.project = f"unlabeled_project_{datetime.now().isoformat()}"
        return self
