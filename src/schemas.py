"""Module containing output schemas for PYF."""

from typing import Annotated

from pydantic import BaseModel, Field


class Point(BaseModel):
    x: Annotated[float, Field(description="x coordinates of the point")]
    y: Annotated[float, Field(description="y coordinates of the point")]


class Line(BaseModel):
    start: Annotated[Point, Field(description="Starting point of the line")]
    end: Annotated[Point, Field(description="End point of the line")]


class V2FootPrediction(BaseModel):
    arch_highest_point: Annotated[Point, Field(description="Coordinates of highest point on foot arch")]
    arch_height: Annotated[float, Field(description="Height of the foot arch")]
    foot_length: Annotated[float, Field(description="Length of the foot")]
    ground_line: Annotated[Line, Field(description="Ground line on which the foot is")]


class ErrorModel(BaseModel):
    error: Annotated[str, Field(description="Error description, if there is one")]

class V2OutputSchema(BaseModel):
    left_foot: Annotated[V2FootPrediction | ErrorModel, Field(description="Left foot prediction")]
    right_foot: Annotated[V2FootPrediction | ErrorModel, Field(description="Right foot prediction")]
