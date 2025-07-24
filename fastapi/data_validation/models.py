from pydantic import BaseModel, validator
from datetime import datetime
from typing import Optional

class OpenStackLogEntry(BaseModel):
    timestamp: datetime
    pid: int
    level: str
    module: str
    request_id: Optional[str]
    message: str

    @validator("level")
    def check_level(cls, v):
        return v
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL","TRACE"}
        if v not in allowed:
            raise ValueError(f"Invalid log level: {v}")
        return v