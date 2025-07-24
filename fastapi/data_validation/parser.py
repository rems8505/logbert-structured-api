import re
from typing import Optional
from pydantic import BaseModel, validator

class ParsedLog(BaseModel):
    timestamp: str
    pid: str
    level: str
    module: str
    request_id: Optional[str] = None
    message: str

    @validator("timestamp")
    def validate_timestamp(cls, v):
        if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?$", v):
            raise ValueError("Invalid timestamp format")
        return v

    @validator("pid")
    def validate_pid(cls, v):
        if not v.isdigit():
            raise ValueError("PID must be numeric")
        return v

    @validator("level")
    def validate_level(cls, v):
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Invalid log level")
        return v

log_pattern = re.compile(
    r"""
    ^(?P<timestamp>\d{4}-\d{2}-\d{2}[\sT]+\d{2}:\d{2}:\d{2}(?:\.\d+)?)
    \s+(?P<pid>\d+)
    \s+(?P<level>[A-Z]+)
    \s+(?P<module>[a-zA-Z0-9_.]+)
    (?:\s+\[(?P<request_id>[^\]]+)\])?
    \s+(?P<message>.+)
    """,
    re.VERBOSE
)

def parse_log_line(log_line: str) -> ParsedLog:
    match = log_pattern.match(log_line.strip())
    if not match:
        raise ValueError("Invalid OpenStack log format: Parsing failed")
    return ParsedLog(**match.groupdict())