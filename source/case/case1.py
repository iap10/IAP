# multithread_yolo_headpose.py
import cv2
import threading
import queue
import time
import ctypes
import os
import shutil
import numpy as np
from detect import HeadPoseEstimate
from yoloc import YoLov7TRT
from collections import deque

# ==== CONFIGURATION ====
PLUGIN_LIBRARY = "/home/iap/final/416100.so"
ENGINE_FILE_PATH = "/home/iap/final/yolov7_416100.engine"
HEADPOSE_ENGINE_PATH = "/home/iap/final/mobilenet_fp16.trt"

categories = ["face", "hand", "paper", "pen", "phone"]

# ==== GLOBALS ====
stop_event = threading.Event()
frame_queue = queue.Queue(maxsize=5)
detection_queue = queue.Queue(maxsize=5)
visual_queue = queue.Queue(maxsize=5)
drowsiness_pose_history = deque(maxlen=30)
state_window = deque(maxlen=60)
focus_ratio = 0.0

# Head pose MAC accumulation
reference_pose = None
i = 0
yaw_mac, pitch_mac, roll_mac = 0.0, 0.0, 0.0


def classify_state(head_ok, sleep, iphone):
    return 0 if iphone == 1 or sleep == 1 or head_ok == 0 else 1
# ==== FUNCTION ====

def detect_drowsiness_combined(yaw, pitch, roll,
                               movement_threshold=3, pitch_threshold=30, duration=30):
    """고개 숙인 채 움직임 거의 없으면 졸음으로 판단"""
    drowsiness_pose_history.append((yaw, pitch, roll))

    if len(drowsiness_pose_history) < duration:
        return False

    dyaw = np.std([p[0] for p in drowsiness_pose_history])
    dpitch = np.std([p[1] for p in drowsiness_pose_history])
    droll = np.std([p[2] for p in drowsiness_pose_history])

    low_motion = (dyaw < movement_threshold and dpitch < movement_threshold and droll < movement_threshold)
    head_down = pitch > pitch_threshold

    return low_motion and head_down

def judge_final_state(current_frame_state, sleep, window_size=60, threshold=0.8):
    state_window.append(current_frame_state)
    if len(state_window) < window_size:
        return "Calibrating", (0, 255, 0), 0

    focused_ratio = sum(state_window) / len(state_window)
    if focused_ratio >= threshold:
        return "Focused", (0, 255, 0), focused_ratio
    else:
        if sleep == 1:
            return "Sleeping", (255, 0, 0), focused_ratio
        return "Not Focused", (0, 0, 255), focused_ratio

# ==== THREADS ====

def capture_thread_func():
    cap = cv2.VideoCapture(0)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (416, 416))
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)
    cap.release()

def yolo_thread_func(yolo_wrapper):
    frame_count = 0
    last_boxes, last_scores, last_class_ids = None, None, None
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if frame_count % 2 == 0:
            start = time.time()
            boxes, scores, class_ids = yolo_wrapper.infer(frame)
            last_boxes, last_scores, last_class_ids = boxes, scores, class_ids = boxes, scores, class_ids
            yolo_ran = True
        else:
            boxes, scores, class_ids = last_boxes, last_scores, last_class_ids
            yolo_ran = False

        frame_count += 1
        #boxes, scores, class_ids = yolo_wrapper.infer(img_re)
        #print("YOLO Inference Time:", time.time() - start)
        if detection_queue.full():
            try:
                detection_queue.get_nowait()
            except queue.Empty:
                pass
        detection_queue.put((frame, boxes, scores, class_ids, yolo_ran))
        #detection_queue.put((img_re, boxes, scores, class_ids))

def headpose_thread_func(headpose_estimator):
    global reference_pose, i, yaw_mac, pitch_mac, roll_mac, focus_ratio

    while not stop_event.is_set():
        try:
            frame, boxes, scores, class_ids, yolo_ran = detection_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if not yolo_ran:
            continue
        head_ok, sleep, iphone = 1, 0, 0
        found_face = False

        for idx in range(len(class_ids)):
            x1, y1, x2, y2 = map(int, boxes[idx])
            if categories[int(class_ids[idx])] == "face":
                found_face = True
                x0, y0 = x1, y2
                crop = frame[y1:y2, x1:x2]
                #print(f"Crop shape: {crop.shape}")
                try:
                    start = time.time()
                    yaw, pitch, roll = headpose_estimator.infer(crop)
                    sleep = detect_drowsiness_combined(yaw, pitch, roll)
                    #print(f"Yaw: {yaw:.2f}, Pitch: {pitch:.2f}, Roll: {roll:.2f}")
                    #print("Headpose Inference Time:", time.time() - start)
                    if reference_pose is None:
                        yaw_mac += yaw
                        pitch_mac += pitch
                        roll_mac += roll
                        i += 1
                        if i == 30:
                            reference_pose = {
                                'yaw': yaw_mac / 30,
                                'pitch': pitch_mac / 30,
                                'roll': roll_mac / 30
                            }
                            print("[Calibrated] Reference pose set:", reference_pose)

                    if reference_pose is not None:
                        delta_yaw = abs(yaw - reference_pose['yaw'])
                        delta_pitch = abs(pitch - reference_pose['pitch'])
                        delta_roll = abs(roll - reference_pose['roll'])
                        head_ok = delta_yaw <= 25 and delta_pitch <= 30 and delta_roll <= 30
                        color = (0, 255, 0) if head_ok else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    #label = f"Pose(ypr): {int(yaw)},{int(pitch)},{int(roll)}"
                    #cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception as e:
                    print("Head pose failed:", e)
            else:
                label = categories[int(class_ids[idx])]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                if int(class_ids[idx]) == 4:
                    iphone = 1
                
        #head_ok/sleep/class_ids 정보 가능
        if found_face:
            state = classify_state(head_ok, sleep, iphone)
            output_text, color, focus_ratio = judge_final_state(state, sleep, window_size=60, threshold=0.8)
            cv2.putText(frame, output_text, (x0, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if visual_queue.full():
            try:
                visual_queue.get_nowait()
            except queue.Empty:
                pass
        visual_queue.put((frame, None))

def display_thread_func():
    fps_calc_start_time = time.time()
    frame_count = 0
    total_frames = 0
    total_time = 0.0

    while not stop_event.is_set():
        try:
            frame, _ = visual_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_count += 1
        total_frames += 1

        if frame_count % 30 == 0:
            now = time.time()
            elapsed = now - fps_calc_start_time
            fps = 30 / elapsed
            total_time += elapsed
            avg_fps = total_frames / total_time if total_time > 0 else 0.0
            print(f"[FPS] {fps:.2f} | [Avg FPS] {avg_fps:.2f}")
            fps_calc_start_time = now

        cv2.putText(frame, f"Focus Ratio: {focus_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("YOLO + HeadPose", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()



# ==== MAIN ====
def main():
    ctypes.CDLL(PLUGIN_LIBRARY, mode=ctypes.RTLD_GLOBAL)
    if os.path.exists("output/"):
        shutil.rmtree("output/")
    os.makedirs("output/")

    headpose_estimator = HeadPoseEstimate(engine_path=HEADPOSE_ENGINE_PATH)
    yolo_wrapper = YoLov7TRT(ENGINE_FILE_PATH)

    try:
        threads = [
            threading.Thread(target=capture_thread_func, daemon=True),
            threading.Thread(target=yolo_thread_func, args=(yolo_wrapper,), daemon=True),
            threading.Thread(target=headpose_thread_func, args=(headpose_estimator,), daemon=True),
            threading.Thread(target=display_thread_func, daemon=True)
        ]

        for t in threads:
            t.start()
        threads[-1].join()  # display_thread 종료 시 전체 종료

    finally:
        yolo_wrapper.destroy()

if __name__ == "__main__":
    main()
