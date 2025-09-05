from typing import List, Dict, Optional
import numpy as np
from insightface.app import FaceAnalysis


class PersonRecognition:
    """
    Lightweight, real-time person recognition using cosine similarity
    against per-person centroids. Keeps a running centroid per ID and
    caps total embedding count for stability in long sessions.
    """

    def __init__(self, threshold: float = 0.6, max_embeddings: int = 36000):
        # Threshold is cosine distance (1 - dot), smaller is more similar.
        self.threshold = float(threshold)
        self.max_embeddings = int(max_embeddings)
        self.centroids: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self._next_id = 1

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-12)

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # a and b expected normalized
        return float(1.0 - np.dot(a, b))

    def _ensure_capacity(self):
        total = sum(self.counts.values())
        if total <= self.max_embeddings:
            return
        # Down-weight counts to retain stability without heavy pruning.
        for k in list(self.counts.keys()):
            self.counts[k] = max(1, self.counts[k] // 2)

    def match(self, embedding: np.ndarray) -> Optional[str]:
        if not self.centroids:
            return None
        e = self._normalize(embedding.astype(np.float32))
        best_id = None
        best_dist = 1e9
        for pid, c in self.centroids.items():
            d = self._cosine_distance(e, c)
            if d < best_dist:
                best_dist, best_id = d, pid
        if best_dist <= self.threshold:
            return best_id
        return None

    def update(self, embedding: np.ndarray, pid: Optional[str] = None) -> str:
        e = self._normalize(embedding.astype(np.float32))
        if pid is None:
            pid = f"person_{self._next_id}"
            self._next_id += 1

        if pid in self.centroids:
            n = self.counts.get(pid, 1)
            c = self.centroids[pid]
            # Online mean update, then renormalize
            new_c = (c * n + e) / (n + 1)
            self.centroids[pid] = self._normalize(new_c)
            self.counts[pid] = n + 1
        else:
            self.centroids[pid] = e
            self.counts[pid] = 1

        self._ensure_capacity()
        return pid

class FaceWorker:
    def __init__(
        self,
        use_gpu: bool = True,
        det_size=(640, 640),
        providers=None,
        enable_person_recognition: bool = False,
        recognition_threshold: float = 0.6,
        max_embeddings: int = 36000,
    ):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

        self.enable_person_recognition = bool(enable_person_recognition)
        self.recognizer = (
            PersonRecognition(threshold=recognition_threshold, max_embeddings=max_embeddings)
            if self.enable_person_recognition
            else None
        )

    def infer(self, frame) -> List[Dict]:
        out: List[Dict] = []
        faces = self.app.get(frame)
        for f in faces:
            bbox = tuple(int(v) for v in f.bbox)
            kps = f.kps.astype(int).tolist() if getattr(f, "kps", None) is not None else []
            item: Dict = {"bbox": bbox, "kps": kps}

            if self.enable_person_recognition and hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                emb = np.asarray(f.normed_embedding, dtype=np.float32)
                match_id = self.recognizer.match(emb)
                pid = self.recognizer.update(emb, match_id)
                item["person_id"] = pid

            out.append(item)
        return out
