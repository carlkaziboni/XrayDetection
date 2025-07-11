DROP TABLE IF EXISTS xray_detections;

CREATE TABLE xray_detections (
    id SERIAL PRIMARY KEY,
    image_path TEXT NOT NULL,
    findings TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);