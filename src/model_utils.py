"""Module containing utilities functions for ML models."""

from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI
from ultralytics import YOLO

from .utils import (get_arch_height_in_pixels, get_contours_from_prediction,
                    get_length_in_pixels, get_mask_from_contours,
                    get_min_rect_box, get_optimized_foot_box,
                    get_pixel_per_metric)

MODEL_CONF = 0.1
A4_PAPER_SIZE = 297
VERBOSE = False
models = {}


def get_first_step_model(model_path: str = "./weights/first_step_model.pt") -> YOLO:
    return YOLO(model_path)


def get_second_step_pose_model(
    model_path: str = "./weights/second_step_pose_model.pt",
) -> YOLO:
    return YOLO(model_path)


def get_second_step_seg_model(
    model_path: str = "./weights/second_step_seg_model.pt",
) -> YOLO:
    return YOLO(model_path)


def predict_foot_length(
    first_step_model: YOLO, img: np.ndarray, paper_real_size: float = A4_PAPER_SIZE
) -> float:
    """Get foot real length from YOLO prediction."""
    prediction = first_step_model(img, max_det=2, conf=MODEL_CONF, verbose=VERBOSE)[0]

    if len(prediction.masks.xy) != 2:
        raise ValueError("A4 paper or foot is not well visible from the top.")

    temp_foot_contour, paper_contour = get_contours_from_prediction(prediction)
    paper_box = get_min_rect_box(paper_contour)
    img_shape = prediction.orig_img.shape[:2]
    temp_foot_mask = get_mask_from_contours(temp_foot_contour, img_shape)
    paper_mask = get_mask_from_contours(paper_box, img_shape)
    foot_mask = np.uint8(cv2.bitwise_and(paper_mask, temp_foot_mask))

    # Find contours
    foot_contour, _ = cv2.findContours(
        foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    foot_contour = np.squeeze(foot_contour[0], axis=1)
    foot_box = get_min_rect_box(foot_contour)

    foot_box = get_optimized_foot_box(paper_box, foot_box)

    paper_length = get_length_in_pixels(paper_box)
    foot_length = get_length_in_pixels(foot_box)
    foot_real_length = foot_length * paper_real_size / paper_length

    return foot_real_length


def get_arch_highest_point(second_step_pose_model: YOLO, img: np.ndarray) -> np.ndarray:
    """Get the highest point on foot arch from YOLO prediction."""
    # Predict with the model
    result = second_step_pose_model(img, max_det=1, conf=MODEL_CONF, verbose=VERBOSE)[0]

    # save foot arch points
    arch_point = result.keypoints.xy.cpu().detach().squeeze().numpy()

    if len(arch_point) != 2:
        raise ValueError("Arch has not been found from the front.")

    return arch_point


def get_foot_contour(second_step_seg_model: YOLO, img: np.ndarray) -> np.ndarray:
    """Get foot contour from YOLO prediction."""
    result = second_step_seg_model(img, max_det=1, conf=MODEL_CONF, verbose=VERBOSE)[0]

    prediction = result.masks.xy

    if len(prediction) != 1:
        raise ValueError("Foot is not well visible from the front.")

    # save foot contour
    foot_contour = prediction[0]
    return foot_contour


def get_foot_arch_height(
    arch_point: np.ndarray, foot_contour: np.ndarray, foot_length: float
) -> float:
    """Get foot arch height."""
    min_rect_box = get_min_rect_box(foot_contour)
    pixel_per_metric = get_pixel_per_metric(min_rect_box, foot_length)
    height_in_pixels = get_arch_height_in_pixels(arch_point, min_rect_box)
    return height_in_pixels / pixel_per_metric


def predict_foot_arch(
    second_step_pose_model: YOLO,
    second_step_seg_model: YOLO,
    img: np.ndarray,
    foot_length: float,
) -> float:
    """Predict foot arch."""
    highest_point = get_arch_highest_point(second_step_pose_model, img)

    foot_contour = get_foot_contour(second_step_seg_model, img)

    bbox = get_min_rect_box(foot_contour)
    pixel_per_metric = get_pixel_per_metric(bbox, foot_length)
    height_in_pixels = get_arch_height_in_pixels(highest_point, bbox)

    arch_height = height_in_pixels / pixel_per_metric

    return highest_point, arch_height, [bbox[3].tolist(), bbox[2].tolist()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    models["first_step_model"] = get_first_step_model()
    models["second_step_pose_model"] = get_second_step_pose_model()
    models["second_step_seg_model"] = get_second_step_seg_model()

    yield

    del models["first_step_model"]
    del models["second_step_pose_model"]
    del models["second_step_seg_model"]
