"""Module containing v2 code."""

from typing import Annotated

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from src.schemas import V2OutputSchema

from ..model_utils import predict_foot_arch, predict_foot_length, models
from ..utils import NB_IMAGES, get_images, convert_numpy_point_to_dict

v2_router = APIRouter(prefix="/v2", tags=["v2"])


@v2_router.post("/compute_feet_measurements")
@v2_router.post("/api/compute_feet_measurements")
async def predict(images: Annotated[list[UploadFile], File(...)]) -> V2OutputSchema:
    if not images or len(images) != NB_IMAGES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Exactly 4 images are required")

    try:
        organized_images = await get_images(images)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    response = {}
    for foot_side, imgs in organized_images.items():
        try:
            foot_length = predict_foot_length(models["first_step_model"], imgs["top"])
            highest_point, arch_height, ground_line = predict_foot_arch(
                models["second_step_pose_model"],
                models["second_step_seg_model"],
                imgs["front"],
                foot_length,
            )
            response[foot_side] = {
                "arch_highest_point": convert_numpy_point_to_dict(highest_point),
                "arch_height": arch_height,
                "foot_length": foot_length,
                "ground_line": {
                    "start": convert_numpy_point_to_dict(ground_line[0]),
                    "end": convert_numpy_point_to_dict(ground_line[1]),
                },
            }
        except ValueError as exc:
            response[foot_side] = {"error": str(exc)}

    return response
