![Logo](zen.png)

# Multi-Camera License Plate Recognition

FastAPI service for synchronized multi-camera license plate detection + OCR, with MinIO image upload and WebSocket result streaming.

## What It Does

- Reads multiple RTSP/video sources from `configs/config.yaml`
- Synchronizes frames across cameras (`src/frame_gather.py`)
- Detects plates with YOLO (`src/detector_yolo.py`)
- Recognizes text with ONNX RobustScanner (`src/recognizer.py`)
- Filters detections by confidence (`0.94` in `src/aysnc_server.py`)
- Draws boxes/text and uploads annotated frames to MinIO
- Streams result payloads over WebSocket (MsgPack)
- Serves API docs and a static dashboard page

## Repository Layout

- `plate_server.py`: FastAPI app + API endpoints
- `src/aysnc_server.py`: async processing service (`LicensePlateServer`)
- `src/frame_gather.py`: synchronized frame capture
- `src/run_yolo.py`: detector + recognizer pipeline
- `src/detector_yolo.py`: Ultralytics YOLO wrapper
- `src/recognizer.py`: OCR pipeline
- `src/minio_uplaoder.py`: MinIO clients
- `configs/config.yaml`: runtime config
- `dashboard/index.html`: static dashboard
- `docker-compose.yml`: app + MinIO

## Requirements

- Python 3.10+
- Linux/macOS recommended for OpenCV/video tooling
- Model files present:
  - `models/yolo12n_openvino_model/`
  - `models/rb_scaner.onnx`

## Local Run

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install openvino msgpack
pip install -r requirements.txt
```

2. Start API server:

```bash
python plate_server.py
```

Server runs on `0.0.0.0:8003`.

## Docker Run

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8003`
- Swagger UI: `http://localhost:8003/docs`
- Dashboard page: `http://localhost:8003/dashboard`
- MinIO API: `http://localhost:9000`
- MinIO console: `http://localhost:9001`

## Configuration

Main config file: `configs/config.yaml`.

Current expected fields include:

```yaml
rtsp_urls:
  - id: 11
    url: /path/to/source1.mp4
  - id: 12
    url: rtsp://user:pass@camera/stream

log_level: INFO
endpoint: 0.0.0.0:9003
access_key: minioadmin
secret_key: minioadmin
bucket: detector-frames
dashboard_ws: ws://<dashboard-host>:8075
upload_interval: 5
```

Notes:
- `rtsp_urls` is required for frame ingestion.
- The service currently initializes MinIO with hardcoded endpoint/credentials in `src/aysnc_server.py` (`0.0.0.0:9000`, `minioadmin/minioadmin`) and uses `bucket` from YAML.

## API

### `POST /config`
Create/replace config file (JSON or YAML body). Refuses update while service is running.

### `PUT /config`
Alias of `POST /config` in current implementation.

### `POST /start`
Start frame sync + processing + websocket server.

Body (`UploaderConfig`):

```json
{
  "upload_interval": 5.0,
  "download_verify": false,
  "download_dir": "downloaded",
  "target_width": 640
}
```

### `POST /stop`
Stop background processing.

### `GET /health`
Health + running flag.

### `GET /get-config`
Returns loaded YAML config + runtime config.

## WebSocket Streams

- Internal result stream: `ws://<host>:9101`
  - Broadcast from `LicensePlateServer`
  - Payload format: MsgPack
  - Example top-level shape:

```json
{
  "results": [
    {
      "cam_id": 11,
      "ocr": "12A34567",
      "confidence": 0.97,
      "minio_path": "cam_11/2026-02-22/1740....jpg",
      "bbox": [100, 200, 260, 260]
    }
  ]
}
```

- External dashboard forward target: `dashboard_ws` from config
  - Same packed payload is forwarded when available.

## Quick API Examples

Write config:

```bash
curl -X POST http://localhost:8003/config \
  -H "Content-Type: application/json" \
  -d '{
    "rtsp_urls": [
      { "id": 11, "url": "/home/arshia/Downloads/parking-1.mp4" },
      { "id": 12, "url": "/home/arshia/Downloads/video.mp4" }
    ],
    "log_level": "INFO",
    "endpoint": "0.0.0.0:9003",
    "access_key": "minioadmin",
    "secret_key": "minioadmin",
    "bucket": "detector-frames",
    "dashboard_ws": "ws://192.168.7.119:8075",
    "upload_interval": 5
  }'
```

Start processing:

```bash
curl -X POST http://localhost:8003/start \
  -H "Content-Type: application/json" \
  -d '{"upload_interval": 5.0, "download_verify": false}'
```

Stop processing:

```bash
curl -X POST http://localhost:8003/stop
```

## Output and Artifacts

- Cropped plates: `cropped_plates/`
- Optional verification downloads: `downloaded/` (when enabled)
- Optional synchronized tile snapshots: path from `output_dir` in config

## Known Caveats

- Filename typo in source is intentional in current repo: `src/aysnc_server.py`.
- MinIO env vars in `docker-compose.yml` use `MINIO_ENDPOINT=minio:9001`, but MinIO object API is usually `9000`.
- Swagger endpoint is custom-mounted at `/docs`.

## Troubleshooting

- Service won’t start:
  - Ensure model files exist in `models/`
  - Check camera/video paths in `configs/config.yaml`
  - Verify MinIO is reachable and bucket is valid
- No detections:
  - Lower detection threshold in detector flow if needed
  - Confirm input frames are non-black and readable
- No dashboard updates:
  - Confirm `dashboard_ws` is reachable
  - Check WebSocket clients are connected to `:9101` if using internal stream
