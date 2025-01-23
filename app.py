from flask import Flask, jsonify, request
from flask_cors import CORS

from model_utils import (get_first_step_model, get_second_step_pose_model,
                         get_second_step_seg_model, predict_foot_arch,
                         predict_foot_length)
from utils import NB_IMAGES, get_images, MAP_SRC_NAME_TO_DEST_NAME


app = Flask(__name__)
CORS(app)


first_step_model = get_first_step_model()
second_step_pose_model = get_second_step_pose_model()
second_step_seg_model = get_second_step_seg_model()


@app.route("/api")
def hello_world():
    return "<p>Hello on Print Your Feet!</p>"


@app.route("/api/compute_feet_measurements", methods=["POST"])
def predict():
    if "images" not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist("images")

    if len(files) != NB_IMAGES:
        return jsonify({"error": "Exactly 4 images are required"}), 400

    response = {}
    try:
        organized_images = get_images(files)
        for foot_side, images in organized_images.items():
            foot_length = predict_foot_length(first_step_model, images["top"])
            highest_point, arch_height, ground_line = predict_foot_arch(
                second_step_pose_model,
                second_step_seg_model,
                images["front"],
                foot_length,
            )
            response[foot_side] = {
                "arch_highest_point": highest_point.tolist(),
                "arch_height": arch_height,
                "foot_length": foot_length,
                "ground_line": ground_line,
            }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(response), 200


@app.route("/api/compute_feet_measurements_bubble", methods=["POST"])
def predict_bubble():
    files = [request.files.get(file_key) for file_key in MAP_SRC_NAME_TO_DEST_NAME.keys()]

    if None in files or len(files) != NB_IMAGES:
        return jsonify({"error": "Exactly 4 images are required"}), 400

    response = {}
    try:
        organized_images = get_images(files)
        for foot_side, images in organized_images.items():
            foot_length = predict_foot_length(first_step_model, images["top"])
            highest_point, arch_height, ground_line = predict_foot_arch(
                second_step_pose_model,
                second_step_seg_model,
                images["front"],
                foot_length,
            )
            response[foot_side] = {
                "arch_highest_point": highest_point.tolist(),
                "arch_height": arch_height,
                "foot_length": foot_length,
                "ground_line": ground_line,
            }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(response), 200
