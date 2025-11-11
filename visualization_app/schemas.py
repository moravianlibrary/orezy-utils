from pydantic import BaseModel, Field


class Page(BaseModel):
    xc: float
    yc: float
    width: float
    height: float
    confidence: float = 0
    angle: float = 0
    flags: list[str] = Field(default_factory=list)
    type: str | None = None


class Scan(BaseModel):
    filename: str
    angle: float = 0
    predicted_pages: list[Page] = Field(default_factory=list)
