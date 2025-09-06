"""Module containing v1 code."""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..model_utils import models, predict_foot_arch, predict_foot_length
from ..utils import NB_IMAGES, get_images

v1_router = APIRouter(tags=["v1"])


@v1_router.post("/compute_feet_measurements")
async def predict(
    images: Annotated[
        list[UploadFile],
        File(
            ...,
            description="List of 4 images named: left_top, right_top, left_front and right_front",
        ),
    ],
):
    if not images or len(images) != NB_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly 4 images are required",
        )

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
                "arch_highest_point": highest_point.tolist(),
                "arch_height": arch_height,
                "foot_length": foot_length,
                "ground_line": ground_line,
            }
        except ValueError as exc:
            response[foot_side] = {"error": str(exc)}

    return response
