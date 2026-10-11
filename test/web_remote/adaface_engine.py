"""Test-only AdaFace adapter and gallery. No writes to the core database."""
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
import numpy as np
from app.performance import performance
from app.recognition.adaface_embedder import AdaFaceEmbedder

UI_DIR = Path(__file__).resolve().parent
MODEL_PATH = UI_DIR.parents[1] / "models" / "adaface_r100.onnx"
GALLERY_DIR = UI_DIR / "gallery"
MATCH_THRESHOLD = 0.50  # demo-only cosine threshold; not inherited from SFace


class AdaEngine(AdaFaceEmbedder):
    def __init__(self, model_path=MODEL_PATH):
        import onnxruntime as ort
        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError("Missing models/adaface_r100.onnx")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        self.model_id = "adaface:core-preprocess-v1:" + digest.hexdigest()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.lock = threading.Lock()
        performance.event("adaface_ready", providers=self.providers, model_id=self.model_id)

    def analyze(self, frame, detection):
        # Reuse exactly the core alignment/preprocessing, without modifying it.
        with self.lock, performance.measure("embedding_ms"):
            aligned = self._align(frame, detection)
            blob = np.ascontiguousarray(self._preprocess(aligned), dtype=np.float32)
            raw = self.session.run([self.output_name], {self.input_name: blob})[0]
        embedding = np.asarray(raw, np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (512,) or not np.isfinite(embedding).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid AdaFace embedding")
        return embedding / norm, dict(embedding_dim=512, embedding_norm=norm,
                                      normalized_norm=1.0, providers=self.providers,
                                      kps=detection.landmarks.tolist())

class DemoGallery:
    def __init__(self, model_id, directory=GALLERY_DIR):
        self.model_id = model_id
        self.directory = Path(directory)
        self.lock = threading.Lock()

    def load(self):
        people, vectors = [], []
        with self.lock:
            for path in sorted(self.directory.glob("*.npz")):
                try:
                    with np.load(path, allow_pickle=False) as row:
                        meta = json.loads(str(row["metadata"].item()))
                        matrix = np.asarray(row["embeddings"], np.float32)
                    if meta["model_id"] != self.model_id or matrix.ndim != 2 or matrix.shape[1] != 512:
                        continue
                    norms = np.linalg.norm(matrix, axis=1)
                    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms <= 0) or len(meta["poses"]) != len(matrix):
                        continue
                    for pose, vector in zip(meta["poses"], matrix / norms[:, None]):
                        people.append({**meta, "pose": pose})
                        vectors.append(vector)
                except (OSError, ValueError, KeyError):
                    performance.count("gallery_invalid_records")
        return people, np.asarray(vectors, np.float32).reshape(-1, 512)

    def save(self, name, room, embeddings):
        matrix = np.asarray(list(embeddings.values()), np.float32)
        if matrix.shape != (5, 512) or not np.isfinite(matrix).all():
            raise ValueError("Five valid AdaFace samples required")
        if np.any(np.linalg.norm(matrix, axis=1) <= 0):
            raise ValueError("Zero AdaFace sample")
        code = "DEMO-" + uuid.uuid4().hex[:12]
        meta = dict(code=code, name=name, room=room, model_id=self.model_id,
                    poses=list(embeddings))
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            target = self.directory / (code + ".npz")
            temporary = target.with_suffix(".tmp")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, metadata=json.dumps(meta, ensure_ascii=False), embeddings=matrix)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return dict(id=code, code=code, name=name, room=room)
