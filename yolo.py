from typing import List, Dict, Tuple
from ultralytics import YOLO

class YoloWorker:
    def __init__(self, weights: str, use_gpu: bool = True, conf: float = 0.4, iou: float = 0.5):
        self.model = YOLO(weights)
        self.device = 0 if use_gpu else "cpu"
        if hasattr(self.model, "fuse"):
            try:
                self.model.fuse()
            except Exception:
                pass
        try:
            self.model.overrides["conf"] = conf
            self.model.overrides["iou"] = iou
        except Exception:
            pass

    def infer(self, frame) -> List[Dict]:
        """Return list of detections: {bbox:(x1,y1,x2,y2), cls:str, cls_id:int, conf:float}"""
        results = self.model(frame, device=self.device, verbose=False)
        r = results[0]
        dets: List[Dict] = []
        names = getattr(getattr(self.model, "model", self.model), "names", {})
        if getattr(r, "boxes", None) is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cls_id = int(b.cls[0]) if b.cls is not None else -1
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                dets.append({
                    "bbox": (x1, y1, x2, y2),
                    "cls": label,
                    "cls_id": cls_id,
                    "conf": conf,
                })
        return dets