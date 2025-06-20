import cv2
import time
import numpy as np
import ctypes
import os
import shutil
from detect import HeadPoseEstimate
from yoloc import YoLov7TRT
from collections import deque

# ==== CONFIGURATION ====
PLUGIN_LIBRARY = "/home/iap/final/416100.so"
ENGINE_FILE_PATH = "/home/iap/final/yolov7_416100.engine"
HEADPOSE_ENGINE_PATH = "/home/iap/final/mobilenet_fp16.trt"

categories = ["face", "hand", "paper", "pen", "phone"]

reference_pose = None
i = 0
yaw_mac, pitch_mac, roll_mac = 0.0, 0.0, 0.0
focus_ratio = 0.0

state_window = deque(maxlen=60)
drowsiness_pose_history = []
max_drowsiness_len = 30
last_yolo_results = None

def classify_state(head_ok, sleep, iphone):
    return 0 if iphone == 1 or sleep == 1 or head_ok == 0 else 1

def detect_drowsiness_combined(yaw, pitch, roll,
                               movement_threshold=3, pitch_threshold=30, duration=30):
    drowsiness_pose_history.append((yaw, pitch, roll))
    if len(drowsiness_pose_history) > duration:
        drowsiness_pose_history.pop(0)

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
        return "Calibrating", (0, 255, 0), 0.0

    focused_ratio = sum(state_window) / len(state_window)
    if focused_ratio >= threshold:
        return "Focused", (0, 255, 0), focused_ratio
    else:
        if sleep == 1:
            return "Sleeping", (255, 0, 0), focused_ratio
        return "Not Focused", (0, 0, 255), focused_ratio

def main():
    global reference_pose, i, yaw_mac, pitch_mac, roll_mac, focus_ratio

    ctypes.CDLL(PLUGIN_LIBRARY, mode=ctypes.RTLD_GLOBAL)
    if os.path.exists("output/"):
        shutil.rmtree("output/")
    os.makedirs("output/")

    cap = cv2.VideoCapture(0)
    headpose_estimator = HeadPoseEstimate(engine_path=HEADPOSE_ENGINE_PATH)
    yolo_wrapper = YoLov7TRT(ENGINE_FILE_PATH)

    frame_count = 0
    last_yolo_results = None
    prev_time = time.time()

    # FPS & Average FPS 계산용 변수
    total_frame_count = 0
    total_elapsed_time = 0.0
    fps = 0.0
    avg_fps = 0.0
    fps_timer_start = time.time()
    

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                #print("카메라에서 프레임을 읽지 못했습니다.")
                break

            frame = cv2.resize(frame, (416, 416))
            head_ok, sleep, iphone = 1, 0, 0
            found_face = False
            x0, y0 = 10, 10

            if frame_count % 2 == 0:
                boxes, scores, class_ids = yolo_wrapper.infer(frame)
                last_yolo_results = (boxes, scores, class_ids)
            else:
                if last_yolo_results is None or last_yolo_results[0] is None:
                    continue
                boxes, scores, class_ids = last_yolo_results
                for idx, cls_id in enumerate(class_ids):
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    x0, y0 = x1, y2
                    if cls_id == 0:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        try:
                            yaw, pitch, roll = headpose_estimator.infer(crop)
                            sleep = detect_drowsiness_combined(yaw, pitch, roll)
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
                        except Exception as e:
                            print("HeadPose inference failed:", e)

                        state = classify_state(head_ok, sleep, iphone)
                        output_text, color, focus_ratio = judge_final_state(state, sleep)
                        cv2.putText(frame, output_text, (x0, y0 + 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    elif cls_id == 4:
                        iphone = 1
                        label = "phone"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        state = classify_state(head_ok, sleep, iphone)
                        output_text, color, focus_ratio = judge_final_state(state, sleep)
                    else:
                        label = categories[int(cls_id)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # FPS 계산
            frame_count += 1
            total_frame_count += 1

            if frame_count % 30 == 0:
                now = time.time()
                elapsed = now - fps_timer_start
                fps = 30 / elapsed
                total_elapsed_time += elapsed
                avg_fps = total_frame_count / total_elapsed_time if total_elapsed_time > 0 else 0.0
                print(f"[FPS] {fps:.2f} | [Avg FPS] {avg_fps:.2f}")
                fps_timer_start = now
            # 화면 표시
            cv2.putText(frame, f"Focus Ratio: {focus_ratio:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YOLO + HeadPose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                #print("[Recalibration] Resetting reference pose.")
                i = 0
                reference_pose = None
                yaw_mac, pitch_mac, roll_mac = 0.0, 0.0, 0.0
                state_window.clear()
                focus_ratio = 0.0


    finally:
        #print(f"[Final Average FPS] {avg_fps:.2f}")
        cap.release()
        cv2.destroyAllWindows()
        yolo_wrapper.destroy()

if __name__ == "__main__":
    main()
