from typing import List, Optional

from pydantic import BaseModel, Field
from redisvl.utils.utils import create_ulid


class LabeledData(BaseModel):
    id: str = Field(default_factory=lambda: create_ulid())
    query: str
    query_match: Optional[str] = None
    response: List[dict] = []
