from typing import List, Dict
from insightface.app import FaceAnalysis

class FaceWorker:
    def __init__(self, use_gpu: bool = True, det_size=(640, 640), providers=None):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

    def infer(self, frame) -> List[Dict]:
        out: List[Dict] = []
        faces = self.app.get(frame)
        for f in faces:
            bbox = tuple(int(v) for v in f.bbox)
            kps = f.kps.astype(int).tolist() if getattr(f, "kps", None) is not None else []
            out.append({"bbox": bbox, "kps": kps})
        return out