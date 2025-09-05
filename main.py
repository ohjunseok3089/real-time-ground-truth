import argparse, time, os, sys, threading, queue
from dataclasses import dataclass
import cv2
import numpy as np

import constants  # expects VIDEO_DEVICE, USE_GPU, YOLO_MODEL, SHOW_WINDOW, DRAW_YOLO, DRAW_FACE
from yolo import YoloWorker
from face_rt import FaceWorker

@dataclass
class FrameItem:
    fid: int
    img: np.ndarray
    timestamp: float = 0.0


def draw_overlays(frame: np.ndarray, yolo_dets, face_dets, yolo_latency=0.0, face_latency=0.0, total_latency=0.0):
    # YOLO: orange boxes
    if yolo_dets:
        for d in yolo_dets:
            x1, y1, x2, y2 = d["bbox"]
            label = f"{d['cls']} {d['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(frame, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,180,255), 1, cv2.LINE_AA)
    # Faces: green boxes + landmarks + person ID
    if face_dets:
        for f in face_dets:
            x1, y1, x2, y2 = f["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            for (px, py) in f.get("kps", []) or []:
                cv2.circle(frame, (int(px), int(py)), 2, (0,255,0), -1)
            # Draw person ID if available
            person_id = f.get("person_id")
            if person_id:
                cv2.putText(frame, person_id, (x1, max(0, y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw latency information on top-left corner
    h, w = frame.shape[:2]
    y_offset = 30
    cv2.putText(frame, f"YOLO: {yolo_latency*1000:.1f}ms", (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Face: {face_latency*1000:.1f}ms", (10, y_offset + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"E2E: {total_latency*1000:.1f}ms", (10, y_offset + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame


def _safe_put(q: queue.Queue, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            _ = q.get_nowait()  # drop oldest
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=constants.VIDEO_DEVICE)
    ap.add_argument("--weights", default=constants.YOLO_MODEL)
    ap.add_argument("--gpu", action="store_true" if constants.USE_GPU else "store_false")
    ap.add_argument("--size", default="", help="WxH (e.g., 1280x720). Empty = keep input.")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--save", default="", help="Path to MP4 output (optional)")
    ap.add_argument("--stdout-mjpeg", action="store_true", help="Write annotated MJPEG to stdout")
    ap.add_argument("--log-every", type=int, default=60)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.device}. Make sure something is writing into it and you have permission.")

    # Try hinting size
    if args.size:
        try:
            w, h = map(int, args.size.lower().split("x"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        except Exception:
            pass

    ok, probe = cap.read()
    if not ok:
        raise RuntimeError("Failed to read a frame from the device.")
    H, W = probe.shape[:2]

    # Optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (W, H))
        if not writer.isOpened():
            print(f"[WARN] Cannot open writer at {args.save}. Continuing without saving.", file=sys.stderr)
            writer = None

    # Workers
    yolo_worker = YoloWorker(weights=args.weights, use_gpu=constants.USE_GPU)
    face_worker = FaceWorker(
        use_gpu=constants.USE_GPU,
        enable_person_recognition=getattr(constants, "ENABLE_PERSON_RECOGNITION", False),
        recognition_threshold=getattr(constants, "RECOGNITION_THRESHOLD", 0.6),
        max_embeddings=getattr(constants, "MAX_EMBEDDINGS", 36000),
    )

    disp_q: queue.Queue[FrameItem] = queue.Queue(maxsize=2)
    yolo_in: queue.Queue[FrameItem] = queue.Queue(maxsize=1)
    face_in: queue.Queue[FrameItem] = queue.Queue(maxsize=1)

    latest_yolo = {"fid": -1, "dets": [], "latency": 0.0, "timestamp": 0.0}
    latest_face = {"fid": -1, "dets": [], "latency": 0.0, "timestamp": 0.0}
    stop_flag = threading.Event()

    def capture_loop():
        fid = 0
        while not stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.005); continue
            fid += 1
            timestamp = time.time()
            # copy once for display/compose
            _safe_put(disp_q, FrameItem(fid=fid, img=frame.copy(), timestamp=timestamp))
            # send to workers (copy to avoid concurrent writes)
            _safe_put(yolo_in, FrameItem(fid=fid, img=frame.copy(), timestamp=timestamp))
            _safe_put(face_in, FrameItem(fid=fid, img=frame, timestamp=timestamp))  # face can share last copy

    def yolo_loop():
        while not stop_flag.is_set():
            try:
                item = yolo_in.get(timeout=0.1)
            except queue.Empty:
                continue
            start_time = time.time()
            dets = yolo_worker.infer(item.img)
            latency = time.time() - start_time
            latest_yolo["fid"] = item.fid
            latest_yolo["dets"] = dets
            latest_yolo["latency"] = latency
            latest_yolo["timestamp"] = item.timestamp

    def face_loop():
        while not stop_flag.is_set():
            try:
                item = face_in.get(timeout=0.1)
            except queue.Empty:
                continue
            start_time = time.time()
            dets = face_worker.infer(item.img)
            latency = time.time() - start_time
            latest_face["fid"] = item.fid
            latest_face["dets"] = dets
            latest_face["latency"] = latency
            latest_face["timestamp"] = item.timestamp

    def compose_loop():
        n = 0
        t0 = time.time()
        while not stop_flag.is_set():
            try:
                item = disp_q.get(timeout=0.2)
            except queue.Empty:
                continue
            frame = item.img
            
            # Calculate end-to-end latency of results used (age of latest outputs)
            current_time = time.time()
            yolo_age = current_time - latest_yolo["timestamp"] if latest_yolo["timestamp"] > 0 else 0.0
            face_age = current_time - latest_face["timestamp"] if latest_face["timestamp"] > 0 else 0.0
            e2e_latency = max(yolo_age, face_age)
            
            # Use most recent results (asynchronous). It's okay if fids don't match exactly.
            frame = draw_overlays(
                frame,
                latest_yolo["dets"],
                latest_face["dets"],
                latest_yolo["latency"],
                latest_face["latency"],
                e2e_latency,
            )

            if writer:
                writer.write(frame)
            if args.stdout_mjpeg:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    sys.stdout.buffer.write(buf.tobytes())
                    sys.stdout.flush()

            n += 1
            if n % args.log_every == 0:
                fps = n / max(1e-6, (time.time() - t0))
                dy = item.fid - latest_yolo["fid"]
                df = item.fid - latest_face["fid"]
                yolo_lat_ms = latest_yolo["latency"] * 1000
                face_lat_ms = latest_face["latency"] * 1000
                total_lat_ms = e2e_latency * 1000
                print(f"[{n:6d}] FPS={fps:5.1f} | lag(yolo={dy}, face={df}) | latency(yolo={yolo_lat_ms:.1f}ms, face={face_lat_ms:.1f}ms, total={total_lat_ms:.1f}ms)", file=sys.stderr)

    th_cap = threading.Thread(target=capture_loop, daemon=True)
    th_yolo = threading.Thread(target=yolo_loop, daemon=True)
    th_face = threading.Thread(target=face_loop, daemon=True)
    th_cmp = threading.Thread(target=compose_loop, daemon=True)

    print(f"[INFO] Running realtime on {args.device} @ {W}x{H}; save={'off' if not writer else args.save}; stdout_mjpeg={args.stdout_mjpeg}", file=sys.stderr)

    th_cap.start(); th_yolo.start(); th_face.start(); th_cmp.start()

    try:
        while th_cmp.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        cap.release()
        if writer:
            writer.release()
        print("[INFO] Stopped.", file=sys.stderr)


if __name__ == "__main__":
    run()
