import cv2
import threading
import queue
import time
import ctypes
import os
import shutil
import numpy as np
from detectc import HeadPoseEstimate
from yoloc import YoLov7TRT
from collections import deque

# ==== CONFIGURATION ====
PLUGIN_LIBRARY = "/home/iap/final/416100.so"
ENGINE_FILE_PATH = "/home/iap/final/yolov7_416100.engine"
HEADPOSE_ENGINE_PATH = "/home/iap/final/mobilenetv3_small.onnx"
categories = ["face", "hand", "paper", "pen", "phone"]

# ==== GLOBAL ====
stop_event = threading.Event()
yolo_result_queue = queue.Queue(maxsize=5)
drowsiness_pose_history = deque(maxlen=30)
state_window = deque(maxlen=60)
focus_ratio = 0.0
reference_pose = None
i = 0
yaw_mac, pitch_mac, roll_mac = 0.0, 0.0, 0.0

# ==== 상태 판단 함수 ====
def classify_state(head_ok, sleep, iphone):
    return 0 if iphone == 1 or sleep == 1 or head_ok == 0 else 1



def detect_drowsiness_combined(yaw, pitch, roll, reference_pose,
                                      movement_threshold=6, pitch_offset_threshold=12, duration=30):
    drowsiness_pose_history.append((yaw, pitch, roll))

    if len(drowsiness_pose_history) < duration or reference_pose is None:
        return False

    dyaw = np.std([p[0] for p in drowsiness_pose_history])
    dpitch = np.std([p[1] for p in drowsiness_pose_history])
    droll = np.std([p[2] for p in drowsiness_pose_history])
    pitch_delta = reference_pose['pitch'] - pitch
    return (dyaw < movement_threshold and dpitch < movement_threshold and droll < movement_threshold) and pitch_delta > pitch_offset_threshold

def judge_final_state(current_frame_state, sleep, window_size=60, threshold=0.8):
    state_window.append(current_frame_state)
    if len(state_window) < window_size:
        return "Calibrating", (0, 255, 0), 0.0
    focused_ratio = sum(state_window) / len(state_window)
    if focused_ratio >= threshold:
        return "Focused", (0, 255, 0), focused_ratio
    else:
        if sleep == 1:
            return "Sleeping", (255, 0, 0), focused_ratio
        return "Not Focused", (0, 0, 255), focused_ratio

# ==== Thread 1: YOLO Detection (GPU) ====
def yolo_detection_thread(yolo_wrapper):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        #print("Camera failed to open.")
        stop_event.set()
        return

    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (416, 416))
        if not ret:
            break

        start = time.time()
        boxes, scores, class_ids = yolo_wrapper.infer(frame)
        #print(f"[YOLO] Frame {frame_count}: Inference time {time.time() - start:.4f} sec")

        frame_count += 1

        if boxes is not None:
            try:
                yolo_result_queue.put((frame, boxes, scores, class_ids), timeout=0.1)
            except queue.Full:
                pass

    cap.release()

# ==== Thread 2: HeadPose + 상태 판단 + Display (CPU or GPU) + FPS 출력 ====
def headpose_and_display_thread(headpose_estimator):
    global reference_pose, i, yaw_mac, pitch_mac, roll_mac, focus_ratio


    last_head_result = (1, 0, 0)  # head_ok, sleep, iphone

    while not stop_event.is_set():
        try:
            frame, boxes, scores, class_ids = yolo_result_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        head_ok, sleep, iphone = last_head_result
        head_ok, sleep, iphone = 1, 0, 0

        for idx, cls_id in enumerate(class_ids):
            x1, y1, x2, y2 = map(int, boxes[idx])
            label = categories[int(cls_id)]

            if label == "face":
                x0, y0 = x1, y2
                crop = frame[y1:y2, x1:x2]

                try:
                    yaw, pitch, roll = headpose_estimator.infer(crop)
                    sleep = detect_drowsiness_combined(yaw, pitch, roll, reference_pose)
                    
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
                    else:
                        delta_yaw = abs(yaw - reference_pose['yaw'])
                        delta_pitch = abs(pitch - reference_pose['pitch'])
                        delta_roll = abs(roll - reference_pose['roll'])
                        head_ok = delta_yaw <= 25 and delta_pitch <= 30 and delta_roll <= 30
                        color = (0, 255, 0) if head_ok else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    state = classify_state(head_ok, sleep, iphone)
                    output_text, color, focus_ratio = judge_final_state(state, sleep)
                    cv2.putText(frame, output_text, (x0, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                except Exception as e:
                    print("Head pose failed:", e)

            elif label == "phone":
                iphone = 1
                x0, y0 = x1, y2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                state = classify_state(head_ok, sleep, iphone)
                output_text, color, focus_ratio = judge_final_state(state, sleep)

            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

        last_head_result = (head_ok, sleep, iphone)


        cv2.putText(frame, f"Focus Ratio: {focus_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("YOLO + HeadPose", frame)
        key = cv2.waitKey(1) & 0XFF
        if key == ord('q'):
            stop_event.set()
            break
        elif key == ord('r'):
            #print("[Recalibration] Resetting reference pose.")
            i = 0
            reference_pose = None
            yaw_mac, pitch_mac, roll_mac = 0.0, 0.0, 0.0
            state_window.clear()
            focus_ratio = 0.0
    cv2.destroyAllWindows()

# ==== MAIN ====
def main():
    ctypes.CDLL(PLUGIN_LIBRARY, mode=ctypes.RTLD_GLOBAL)
    if os.path.exists("output/"):
        shutil.rmtree("output/")
    os.makedirs("output/")

    headpose_estimator = HeadPoseEstimate(HEADPOSE_ENGINE_PATH)
    yolo_wrapper = YoLov7TRT(ENGINE_FILE_PATH)

    try:
        threads = [
            threading.Thread(target=yolo_detection_thread, args=(yolo_wrapper,), daemon=True),
            threading.Thread(target=headpose_and_display_thread, args=(headpose_estimator,), daemon=True)
        ]

        for t in threads:
            t.start()

        threads[-1].join()  # display thread 종료 시 전체 종료
    finally:
        yolo_wrapper.destroy()

if __name__ == "__main__":
    main()
