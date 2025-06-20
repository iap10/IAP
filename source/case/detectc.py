import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms
from utils import (
    compute_rotation_matrix_from_ortho6d,
    compute_euler_angles_from_rotation_matrices,
)

class HeadPoseEstimate:
    def __init__(self, onnx_path):
        # CPU 전용 실행 세션
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        # ONNX 모델 입력/출력 이름 획득
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 이미지 전처리 transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image).unsqueeze(0).numpy().astype(np.float32)
        return tensor

    def infer(self, img):
        if img is None or img.size == 0:
            return None, None, None

        input_tensor = self.preprocess(img)

        # ONNX 추론
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        out_arr = outputs[0].flatten()

        # 출력 파싱 및 회전행렬 계산
        if out_arr.size == 6:
            import torch
            rot6d = torch.tensor(out_arr.reshape(1, 6))
            rot_mat = compute_rotation_matrix_from_ortho6d(rot6d)
        elif out_arr.size == 9:
            import torch
            rot_mat = torch.tensor(out_arr.reshape(1, 3, 3))
        else:
            raise ValueError(f"Unexpected output size: {out_arr.size}")

        # Euler 각도 변환
        yaw, pitch, roll = np.degrees(
            compute_euler_angles_from_rotation_matrices(rot_mat)
        )[0, [1, 0, 2]].tolist()

        return yaw, pitch, roll
