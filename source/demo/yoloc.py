import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt
import cv2

CONF_THRESH = 0.25
IOU_THRESHOLD = 0.45

class YoLov7TRT:
    def __init__(self, engine_file_path):
        self.ctx = cuda.Device(0).make_context()
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        stream = cuda.Stream()

        host_inputs, cuda_inputs = [], []
        host_outputs, cuda_outputs = [], []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def preprocess(self, raw_bgr_image):
        image_raw = raw_bgr_image
        h, w, _ = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        r_w = self.input_w / w
        r_h = self.input_h / h

        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0

        image = cv2.resize(image, (tw, th))
        image = cv2.copyMakeBorder(image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        return np.ascontiguousarray(image), image_raw, h, w

    def infer(self, image):
        self.ctx.push()
        input_image, image_raw, origin_h, origin_w = self.preprocess(image)
        np.copyto(self.host_inputs[0], input_image.ravel())

        cuda.memcpy_htod_async(self.cuda_inputs[0], self.host_inputs[0], self.stream)
        self.context.execute_async(batch_size=self.batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0], self.stream)
        self.stream.synchronize()

        output = self.host_outputs[0]
        boxes, scores, class_ids = self.postprocess(output, origin_h, origin_w)
        self.ctx.pop()
        return boxes, scores, class_ids

    def postprocess(self, output, origin_h, origin_w):
        num = int(output[0])
        if num == 0 or len(output) <= 1:
            return np.array([]), np.array([]), np.array([])

        try:
            pred = np.reshape(output[1:], (-1, 6))[:num, :]
        except:
            return np.array([]), np.array([]), np.array([])

        pred = self.keep_largest_per_class(pred, area_thresh=1000)
        boxes = self.non_max_suppression(pred, origin_h, origin_w, CONF_THRESH, IOU_THRESHOLD)

        if boxes.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])

        return boxes[:, :4], boxes[:, 4], boxes[:, 5]

    def keep_largest_per_class(self, pred, area_thresh=1000):
        if pred.shape[0] == 0:
            return np.empty((0, 6))

        filtered = []
        class_ids = np.unique(pred[:, 5])
        for cid in class_ids:
            class_thresh = 2000 if cid == 0 else area_thresh
            class_preds = pred[pred[:, 5] == cid]
            areas = class_preds[:, 2] * class_preds[:, 3]
            class_preds = class_preds[areas >= class_thresh]
            if class_preds.shape[0] == 0:
                continue
            best_idx = np.argmax(class_preds[:, 4])
            filtered.append(class_preds[best_idx])

        return np.array(filtered)

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        if prediction.ndim != 2 or prediction.shape[1] < 6:
            return np.empty((0, 6))

        boxes = prediction[prediction[:, 4] >= conf_thres]
        if boxes.shape[0] == 0:
            return np.empty((0, 6))

        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)

        confs = boxes[:, 4]
        boxes = boxes[np.argsort(-confs)]

        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4], x1y1x2y2=True) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            invalid = large_overlap & label_match
            keep_boxes.append(boxes[0])
            boxes = boxes[~invalid]

        return np.stack(keep_boxes, 0) if len(keep_boxes) else np.empty((0, 6))

    def xywh2xyxy(self, origin_h, origin_w, x):
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h
        return y

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if not x1y1x2y2:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * np.clip(
            inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
        return iou

    def destroy(self):
        self.ctx.pop()
