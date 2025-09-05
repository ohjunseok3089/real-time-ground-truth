from typing import List, Dict, Optional
import numpy as np
from insightface.app import FaceAnalysis
from collections import deque
import time

class PersonRecognition:
    def __init__(self, max_embeddings: int = 36000, recognition_threshold: float = 0.6):
        self.max_embeddings = max_embeddings
        self.recognition_threshold = recognition_threshold
        self.embeddings = deque(maxlen=max_embeddings)
        self.person_ids = deque(maxlen=max_embeddings)
        self.next_person_id = 1
        self.embedding_to_person = {}  # For quick lookup
        
    def add_embedding(self, embedding: np.ndarray, person_id: Optional[str] = None) -> str:
        """Add embedding to gallery and return assigned person_id"""
        if person_id is None:
            person_id = f"person_{self.next_person_id}"
            self.next_person_id += 1
            
        self.embeddings.append(embedding)
        self.person_ids.append(person_id)
        self.embedding_to_person[tuple(embedding)] = person_id
        return person_id
    
    def recognize(self, embedding: np.ndarray) -> str:
        """Recognize person from embedding"""
        if not self.embeddings:
            return self.add_embedding(embedding)
            
        # Calculate cosine similarity (1 - cosine distance)
        similarities = np.dot(list(self.embeddings), embedding)
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        if best_similarity > self.recognition_threshold:
            return self.person_ids[best_match_idx]
        else:
            return self.add_embedding(embedding)

class FaceWorker:
    def __init__(self, use_gpu: bool = True, det_size=(640, 640), providers=None, 
                 enable_person_recognition: bool = True, max_embeddings: int = 36000):
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)
        
        self.enable_person_recognition = enable_person_recognition
        if enable_person_recognition:
            self.person_recognition = PersonRecognition(max_embeddings=max_embeddings)

    def infer(self, frame) -> List[Dict]:
        out: List[Dict] = []
        faces = self.app.get(frame)
        for f in faces:
            bbox = tuple(int(v) for v in f.bbox)
            kps = f.kps.astype(int).tolist() if getattr(f, "kps", None) is not None else []
            
            face_data = {"bbox": bbox, "kps": kps}
            
            if self.enable_person_recognition and hasattr(f, 'normed_embedding'):
                person_id = self.person_recognition.recognize(f.normed_embedding)
                face_data["person_id"] = person_id
                
            out.append(face_data)
        return out