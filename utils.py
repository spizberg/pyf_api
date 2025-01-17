import io

import cv2
import numpy as np
from imutils import perspective
from PIL import Image
from scipy.spatial import distance as dist
from sympy import Line, Point
from werkzeug.utils import secure_filename

NB_IMAGES = 4

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

MAP_SRC_NAME_TO_DEST_NAME = {
    "left_top": "top_left",
    "right_top": "top_right",
    "left_front": "front_left",
    "right_front": "front_right",
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_filename_expected(filename: str) -> bool:
    return filename in MAP_SRC_NAME_TO_DEST_NAME.keys()


def convert_bytes_to_image(image_bytes):
    pil_image = Image.open(io.BytesIO(image_bytes))
    pil_image = pil_image.convert('RGB')
    return np.array(pil_image)[:, :, ::-1]


def organize_input_images(names_to_imgs: dict) -> dict:
    return {
        "left_foot": {
            "top": names_to_imgs["top_left"],
            "front": names_to_imgs["front_left"],
        },
        "right_foot": {
            "top": names_to_imgs["top_right"],
            "front": names_to_imgs["front_right"],
        },
    }


def get_allowed_image(file) -> np.ndarray:
    if not (file and allowed_file(file.filename)):
        raise ValueError("Image format not allowed.")

    filename = secure_filename(file.filename)
    filename_without_ext = filename.split(".")[0].lower()

    if not is_filename_expected(filename_without_ext):
        raise ValueError("The image's name is not recognized.")

    np_img = convert_bytes_to_image(file.read())
    return {MAP_SRC_NAME_TO_DEST_NAME[filename_without_ext]: np_img}


def get_images(files) -> dict[str, np.ndarray]:
    names_to_imgs = {}
    for file in files:
        name_to_img = get_allowed_image(file)
        names_to_imgs.update(name_to_img)

    if len(names_to_imgs) != NB_IMAGES:
        nb_missings = NB_IMAGES - len(names_to_imgs)
        raise ValueError(f"{nb_missings} image(s) is/are missing.")

    organized_input_images = organize_input_images(names_to_imgs)

    return organized_input_images


def get_min_rect_box(contour: np.ndarray) -> np.ndarray:
    """Get minimal rect box."""
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box)
    return box.astype("int")


def sort_contours(list_contours: list[np.ndarray]) -> list[np.ndarray]:
    """Sort (ascendant order) contours in list_contours."""
    return sorted(list_contours, key=cv2.contourArea)


def get_optimized_foot_box(ref_bbox: np.ndarray, obj_bbox: np.ndarray) -> np.ndarray:
    ref_upper_midpoint = midpoint(ref_bbox[0], ref_bbox[1])
    obj_upper_midpoint = midpoint(obj_bbox[0], obj_bbox[1])
    ref_lower_midpoint = midpoint(ref_bbox[2], ref_bbox[3])
    obj_lower_midpoint = midpoint(obj_bbox[2], obj_bbox[3])

    upper_distance = dist.euclidean(ref_upper_midpoint, obj_upper_midpoint)
    lower_distance = dist.euclidean(ref_lower_midpoint, obj_lower_midpoint)

    obj_new_bbox = obj_bbox.copy()
    if upper_distance < lower_distance:
        obj_new_bbox[0] = ref_bbox[0]
        obj_new_bbox[1] = ref_bbox[1]
    else:
        obj_new_bbox[2] = ref_bbox[2]
        obj_new_bbox[3] = ref_bbox[3]
    return obj_new_bbox


def midpoint(point_a, point_b):
    """Calculate the midpoint between two points."""
    return (point_a[0] + point_b[0]) * 0.5, (point_a[1] + point_b[1]) * 0.5


def get_length_in_pixels(bbox: np.ndarray) -> float:
    first_midpoint = midpoint(bbox[0], bbox[1])
    second_midpoint = midpoint(bbox[2], bbox[3])
    return dist.euclidean(first_midpoint, second_midpoint)


def get_pixel_per_metric(bbox: np.ndarray, size: float) -> float:
    euclidian_distance = dist.euclidean(bbox[0], bbox[1])
    return euclidian_distance / size


def get_mask_from_contours(contours: np.ndarray, mask_shape: tuple) -> np.ndarray:
    """Get mask from contours."""
    # Create an empty mask
    output_mask = np.zeros(mask_shape, dtype=np.int32)
    # Fill the contour on the mask
    cv2.fillPoly(output_mask, np.int32([contours]), color=255)
    return output_mask


def get_contours_from_prediction(model_prediction) -> tuple[np.ndarray, np.ndarray]:
    """Get foot and paper contours from YOLO prediction."""
    foot_result, paper_result = None, None
    prediction_summary = model_prediction.summary()
    for prediction in prediction_summary:
        seg = np.array(
            [
                [x, y]
                for x, y in zip(
                    prediction["segments"]["x"], prediction["segments"]["y"]
                )
            ],
            dtype=np.float32,
        )
        if prediction["class"] == 0:
            foot_result = seg
        elif prediction["class"] == 1:
            paper_result = seg
    if foot_result is None or paper_result is None:
        raise ValueError("The A4 paper or the foot is not well visible.")
    return foot_result, paper_result


def get_arch_height_in_pixels(point: np.ndarray, bbox: np.ndarray) -> float:
    highest_point = Point(point)
    point_bl, point_br = Point(bbox[3]), Point(bbox[2])
    bottom_line = Line(point_bl, point_br)
    return float(bottom_line.perpendicular_segment(highest_point).length.evalf())
