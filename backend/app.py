from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from audio_pipeline import (
    AUDIO_EXTENSIONS,
    feature_dimension,
    load_training_manifest,
    predict_speaker,
    train_speaker_identifier,
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
WORKSPACE_DIR = BASE_DIR / "workspace"
UPLOAD_DIR = WORKSPACE_DIR / "uploads"
EXTRACT_DIR = WORKSPACE_DIR / "extracted"
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVE_BYTES
CORS(app)


def ensure_runtime_directories() -> None:
    for directory in (ARTIFACT_DIR, WORKSPACE_DIR, UPLOAD_DIR, EXTRACT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def reset_training_state() -> None:
    shutil.rmtree(ARTIFACT_DIR, ignore_errors=True)
    shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    ensure_runtime_directories()


def allowed_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = destination / member.filename
            resolved_member_path = member_path.resolve()
            if destination_root != resolved_member_path and destination_root not in resolved_member_path.parents:
                raise ValueError("The uploaded archive contains an unsafe file path.")

        archive.extractall(destination)


def json_error(message: str, status_code: int):
    response = jsonify({"message": message})
    response.status_code = status_code
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_: RequestEntityTooLarge):
    return json_error("Upload exceeds the 100 MB limit.", 413)


@app.get("/health")
def healthcheck():
    ensure_runtime_directories()
    return jsonify({"status": "ok", "service": "who-is-speaking-backend"})


@app.post("/train")
def train():
    ensure_runtime_directories()

    uploaded_archive = request.files.get("dataset")
    if uploaded_archive is None or uploaded_archive.filename == "":
        return json_error("Attach a .zip archive of speaker folders before training.", 400)

    archive_name = secure_filename(uploaded_archive.filename)
    if Path(archive_name).suffix.lower() != ".zip":
        return json_error("Training data must be uploaded as a .zip archive.", 400)

    reset_training_state()
    archive_path = UPLOAD_DIR / archive_name
    uploaded_archive.save(archive_path)

    try:
        safe_extract_archive(archive_path, EXTRACT_DIR)
        manifest = train_speaker_identifier(EXTRACT_DIR, ARTIFACT_DIR)
        response_payload = {
            "message": "Training complete. The speaker recognition model is ready.",
            "training_summary": manifest,
            "feature_dimension": feature_dimension(),
            "feature_pipeline": {
                "segmentation": (
                    "Long recordings are split into voiced 3-second windows with 1.5-second "
                    "hop before feature extraction"
                ),
                "time_domain": [
                    "Short-term average energy (mean and standard deviation)",
                    "Zero crossing rate (mean and standard deviation)",
                ],
                "vocal_tract_model": "12th-order LPC coefficients averaged across active frames",
                "spectral_model": "Mean log-mel energy across the first 20 filter banks",
                "normalization": "StandardScaler",
                "classifier": "SVC with rbf kernel and probability=True",
                "open_set_guard": (
                    "60 percent probability threshold plus nearest-neighbor acoustic "
                    "consistency in standardized feature space"
                ),
            },
        }
        return jsonify(response_payload)
    except zipfile.BadZipFile:
        reset_training_state()
        return json_error("The uploaded file is not a valid .zip archive.", 400)
    except ValueError as error:
        reset_training_state()
        return json_error(str(error), 400)
    except Exception as error:
        reset_training_state()
        return json_error(f"Training failed: {error}", 500)


@app.post("/predict")
def predict():
    try:
        load_training_manifest(ARTIFACT_DIR)
    except FileNotFoundError:
        return json_error("Train the model before requesting predictions.", 400)

    uploaded_audio = request.files.get("audio")
    if uploaded_audio is None or uploaded_audio.filename == "":
        return json_error("Attach one .wav, .mp3, or .flac file for prediction.", 400)

    audio_name = secure_filename(uploaded_audio.filename)
    if not allowed_audio_file(audio_name):
        return json_error("Prediction supports only .wav, .mp3, and .flac files.", 400)

    ensure_runtime_directories()
    audio_path = UPLOAD_DIR / audio_name
    uploaded_audio.save(audio_path)

    try:
        prediction = predict_speaker(audio_path, ARTIFACT_DIR)
        return jsonify(prediction)
    except ValueError as error:
        return json_error(str(error), 400)
    except Exception as error:
        return json_error(f"Prediction failed: {error}", 500)
    finally:
        audio_path.unlink(missing_ok=True)


if __name__ == "__main__":
    ensure_runtime_directories()
    app.run(debug=True, host="127.0.0.1", port=5000)
