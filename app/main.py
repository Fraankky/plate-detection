import io
import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.background import BackgroundTask

from .model import MODEL_PATH, model, predict

app = FastAPI(title="Plate Detection API", version="0.2.0")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def _detections(result: Any) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    out: list[dict[str, Any]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(
            {
                "class": result.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return out


def _unlink(p: Path | str) -> None:
    try:
        Path(p).unlink(missing_ok=True)
    except OSError:
        pass


def _h264_path(source: Path) -> Path:
    target = source.with_name(f"{source.stem}_h264.mp4")
    encoders = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    encoder = "libx264" if " libx264 " in encoders else "libopenh264"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-c:v",
                encoder,
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(target),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        _unlink(target)
        raise HTTPException(status_code=500, detail="ffmpeg tidak ditemukan di server") from e
    except subprocess.CalledProcessError as e:
        _unlink(target)
        detail = e.stderr.decode(errors="replace")[-500:]
        raise HTTPException(status_code=500, detail=f"gagal encode H.264: {detail}") from e
    return target


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": str(getattr(model, "ckpt_path", MODEL_PATH))}


# --- foto: upload JPG/PNG -> JSON bbox (frontend gambar bbox di canvas) ---
@app.post("/detect")
async def detect(file: UploadFile = File(..., description="JPG/PNG")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"harus image/*, got {file.content_type}")
    data = await file.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="file kosong")
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"gagal decode image: {e}") from e
    r = predict(np.array(image))  # conf 0.25 default untuk foto
    dets = _detections(r)
    return {"detections": dets, "count": len(dets)}


# --- video: upload video -> annotated MP4 preview (bbox mengikuti plat) ---
@app.post("/video")
async def video(file: UploadFile = File(..., description="MP4")):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"harus video/*, got {file.content_type}")
    suffix = Path(file.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        shutil.copyfileobj(io.BytesIO(await file.read()), tmp_in)
        in_path = Path(tmp_in.name)
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        _unlink(in_path)
        raise HTTPException(status_code=400, detail=f"gagal buka video: {file.filename}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 640)
    w = w if w % 2 == 0 else w + 1
    h = h if h % 2 == 0 else h + 1
    out_fd, out_path_s = tempfile.mkstemp(suffix=".mp4")
    os.close(out_fd)
    out_path = Path(out_path_s)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
    out = cv2.VideoWriter(str(out_path), fourcc, fps if fps > 0 else 5.0, (w, h))
    
    if not out.isOpened():
        cap.release()
        _unlink(in_path)
        _unlink(out_path)
        raise HTTPException(status_code=500, detail="gagal buat VideoWriter mp4")
    try:
        idx = 0
        max_frames = 90  # ponytail: cukup untuk preview
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or idx >= max_frames:
                break
            annotated = predict(frame, conf=0.15).plot()  # ponytail: lower conf untuk plat kecil di video
            if annotated.shape[1] != w or annotated.shape[0] != h:
                annotated = cv2.resize(annotated, (w, h))
            out.write(annotated)
            idx += 1
        if idx == 0:
            raise HTTPException(status_code=400, detail="video tidak ada frame terbaca")
    finally:
        cap.release()
        out.release()
        _unlink(in_path)

    encoded_path = _h264_path(out_path)
    _unlink(out_path)

    def _cleanup() -> None:
        _unlink(encoded_path)

    return FileResponse(
        str(encoded_path),
        media_type="video/mp4",
        filename=f"annotated_{Path(file.filename or 'video').stem}.mp4",
        background=BackgroundTask(_cleanup),
    )
