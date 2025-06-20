import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
from torchvision import transforms
from utils import (
    compute_rotation_matrix_from_ortho6d,
    compute_euler_angles_from_rotation_matrices,
)

class HeadPoseEstimate:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        # CUDA 컨텍스트 생성
        self.ctx = cuda.Device(0).make_context()

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

    def _load_engine(self):
        with open(self.engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]
            )
        ])
        tensor = transform(image).unsqueeze(0)
        return tensor.numpy().astype(np.float32)

    def infer(self, img):
        # 1) 컨텍스트 활성화
        self.ctx.push()

        # 2) 빈 이미지 검사
        if img is None or img.size == 0:
            self.ctx.pop()
            return None, None, None

        # 3) 전처리
        input_tensor = self.preprocess(img).ravel()

        # 4) 바인딩 준비
        inputs, outputs, bindings = [], [], []
        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            host_mem   = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if self.engine.binding_is_input(i):
                np.copyto(host_mem, input_tensor)
                inputs.append((host_mem, device_mem))
            else:
                outputs.append((host_mem, device_mem))
            bindings.append(int(device_mem))

        # 5) 추론 수행
        cuda.memcpy_htod_async(inputs[0][1], inputs[0][0], self.stream)
        self.context.execute_async_v2(bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(outputs[0][0], outputs[0][1], self.stream)
        self.stream.synchronize()

        # 6) 결과 파싱
        out_arr = outputs[0][0]
        '''
        if out_arr is None or out_arr.size < 6:
            self.ctx.pop()
            raise ValueError(f"Invalid output received: {out_arr}")
        '''
        if out_arr.size == 6:
            rot6d = torch.tensor(out_arr.reshape(1,6))
            rot_mat = compute_rotation_matrix_from_ortho6d(rot6d)
        elif out_arr.size == 9:
            rot_mat = torch.tensor(out_arr.reshape(1,3,3))
        else:
            self.ctx.pop()
            raise ValueError(f"Unexpected output size: {out_arr.size}")
        yaw, pitch, roll = np.degrees(
            compute_euler_angles_from_rotation_matrices(rot_mat)
        )[0, [1,0,2]].tolist()

        # 7) 컨텍스트 해제
        self.ctx.pop()

        # 8) 순수 결과 반환
        return yaw, pitch, roll

