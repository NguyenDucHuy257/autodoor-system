"""Complete SCRFD decoding for the test UI; core source is left untouched.
Output geometry follows the InsightFace SCRFD ONNX contract:
https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/scrfd.py
FaceDetection landmarks retain the existing YuNet ordering expected by core consumers.
"""
import threading
import cv2
import numpy as np
from app.recognition.scrfd_detector import SCRFDDetector
from app.recognition.detector import FaceDetection
from app.performance import performance


class WebSCRFDDetector(SCRFDDetector):
    def __init__(self, model_path, input_size=(640,640), score_threshold=.5, nms_threshold=.4):
        super().__init__(model_path, input_size, score_threshold)
        self.nms_threshold = nms_threshold
        self.lock = threading.Lock()
        self.providers = self.session.get_providers()
        self.output_names = [output.name for output in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise ValueError("Web SCRFD requires three stride levels with five landmarks (9 outputs)")
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = (shape[3], shape[2])
        self.centers = {}
        performance.event("scrfd_ready", providers=self.providers, input_size=list(self.input_size), score_threshold=score_threshold)

    def _preprocess(self, frame):
        # SCRFD expects RGB; core adapter currently produces a BGR tensor.
        blob, scale = super()._preprocess(frame)
        return np.ascontiguousarray(blob[:, ::-1]), scale

    def _decode(self, outputs, frame_shape, scale):
        target_width, target_height = self.input_size
        boxes, scores, landmarks = [], [], []
        for level, stride in enumerate((8,16,32)):
            confidence = np.asarray(outputs[level]).reshape(-1)
            distance = np.asarray(outputs[level+3]).reshape(-1,4)*stride
            offsets = np.asarray(outputs[level+6]).reshape(-1,5,2)*stride
            if stride not in self.centers:
                xx, yy = np.meshgrid(np.arange(target_width//stride), np.arange(target_height//stride))
                grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)*stride
                self.centers[stride] = np.repeat(grid, 2, axis=0)
            centers = self.centers[stride]
            if len(confidence) != len(centers) or len(distance) != len(centers) or len(offsets) != len(centers):
                raise ValueError("SCRFD output dimensions do not match input size")
            valid = (confidence >= self.score_threshold) & np.isfinite(confidence) & np.isfinite(distance).all(axis=1) & np.isfinite(offsets).all(axis=(1,2))
            indices = np.flatnonzero(valid)
            if not len(indices): continue
            anchors, delta = centers[indices], distance[indices]
            decoded = np.column_stack((anchors[:,0]-delta[:,0], anchors[:,1]-delta[:,1], anchors[:,0]+delta[:,2], anchors[:,1]+delta[:,3])) / scale
            points = (anchors[:,None,:]+offsets[indices])/scale
            # Reject padding-only boxes; clip only the box, never distort landmarks.
            h, w = frame_shape[:2]
            decoded[:,(0,2)] = np.clip(decoded[:,(0,2)],0,w)
            decoded[:,(1,3)] = np.clip(decoded[:,(1,3)],0,h)
            for box, score, kps in zip(decoded,confidence[indices],points):
                x1,y1,x2,y2 = box
                if x2 <= x1 or y2 <= y1: continue
                boxes.append([float(x1),float(y1),float(x2-x1),float(y2-y1)])
                scores.append(float(score))
                landmarks.append(kps[[1,0,2,4,3]].astype(np.float32))
        if not boxes: return []
        keep = np.asarray(cv2.dnn.NMSBoxes(boxes,scores,self.score_threshold,self.nms_threshold)).reshape(-1)
        results = []
        for at in keep:
            x,y,w,h = boxes[int(at)]
            left,top,right,bottom = int(np.floor(x)),int(np.floor(y)),int(np.ceil(x+w)),int(np.ceil(y+h))
            results.append(FaceDetection(left,top,right-left,bottom-top,landmarks[int(at)],scores[int(at)]))
        return results

    def detect(self, frame):
        if frame is None or frame.size == 0: return []
        with self.lock:
            blob, scale = self._preprocess(frame)
            outputs = self.session.run(self.output_names, {self.input_name:blob})
            return self._decode(outputs, frame.shape, scale)
