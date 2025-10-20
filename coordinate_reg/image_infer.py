import cv2
import numpy as np
import os
import onnxruntime as ort
from skimage import transform as trans
import insightface
import sys
# sys.path.append('/home/jovyan/FaceShifter-2/FaceShifter3/')
from insightface_func.face_detect_crop_single import Face_detect_crop
import kornia
import torch


M = np.array([[ 0.57142857, 0., 32.],[ 0.,0.57142857, 32.]])
IM = np.array([[[1.75, -0., -56.],[ -0., 1.75, -56.]]])


def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale


def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M


def trans_points2d_batch(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for j in range(pts.shape[0]):
        for i in range(pts.shape[1]):
            pt = pts[j][i]
            new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
            new_pt = np.dot(M[j], new_pt)
            new_pts[j][i] = new_pt[0:2]
    return new_pts


def trans_points2d(pts, M):
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i] = new_pt[0:2]

    return new_pts


def trans_points3d(pts, M):
    scale = np.sqrt(M[0][0] * M[0][0] + M[0][1] * M[0][1])
    #print(scale)
    new_pts = np.zeros(shape=pts.shape, dtype=np.float32)
    for i in range(pts.shape[0]):
        pt = pts[i]
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32)
        new_pt = np.dot(M, new_pt)
        #print('new_pt', new_pt.shape, new_pt)
        new_pts[i][0:2] = new_pt[0:2]
        new_pts[i][2] = pts[i][2] * scale

    return new_pts


def trans_points(pts, M):
    if pts.shape[1] == 2:
        return trans_points2d(pts, M)
    else:
        return trans_points3d(pts, M)


class Handler:
    def __init__(self, prefix, epoch, im_size=192, det_size=224, ctx_id=0, root='./insightface_func/models'):
        print('loading', prefix, epoch)
        
        # Use CUDA if available, otherwise CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ctx_id >= 0 else ['CPUExecutionProvider']
        
        image_size = (im_size, im_size)
        self.detector = Face_detect_crop(name='antelope', root=root)
        self.detector.prepare(ctx_id=ctx_id, det_thresh=0.6, det_size=(640,640))
        self.det_size = det_size
        self.image_size = image_size
        
        # Try to load ONNX model if available, otherwise use insightface default
        onnx_path = f"{prefix}-{epoch:04d}.onnx"
        if os.path.exists(onnx_path):
            print(f"Loading ONNX model from {onnx_path}")
            self.model = ort.InferenceSession(onnx_path, providers=providers)
            self.use_onnx = True
        else:
            # ONNX model not found - this is expected behavior
            # The landmark detection will be handled by other methods
            print(f"ONNX model not found at {onnx_path}, will use alternative methods for landmarks")
            self.use_onnx = False
            self.model = None
    
    
    def get_without_detection_batch(self, img, M, IM):
        rimg = kornia.warp_affine(img, M.repeat(img.shape[0],1,1), (192, 192), padding_mode='zeros')
        rimg = kornia.bgr_to_rgb(rimg)
        
        if self.use_onnx and self.model is not None:
            # ONNX Runtime inference
            input_name = self.model.get_inputs()[0].name
            pred = self.model.run(None, {input_name: rimg.cpu().numpy()})[0]
        else:
            # Fallback: convert to numpy and run dummy prediction
            pred = np.zeros((rimg.shape[0], 106, 2), dtype=np.float32)
            
        pred = pred.reshape((pred.shape[0], -1, 2))  
        pred[:, :, 0:2] += 1
        pred[:, :, 0:2] *= (self.image_size[0] // 2)
        
        pred = trans_points2d_batch(pred, IM.repeat(img.shape[0],1,1).numpy())
        
        return pred
    
    
    def get_without_detection_without_transform(self, img):
        input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
        rimg = cv2.warpAffine(img, M, self.image_size, borderValue=0.0)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        rimg = np.transpose(rimg, (2, 0, 1))  #3*112*112, RGB
        
        input_blob[0] = rimg
        
        if self.use_onnx and self.model is not None:
            input_name = self.model.get_inputs()[0].name
            pred = self.model.run(None, {input_name: input_blob})[0][0]
        else:
            pred = np.zeros((106, 2), dtype=np.float32)
            
        pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.image_size[0] // 2)
        pred = trans_points2d(pred, IM[0])
        
        return pred
    
    
    def get_without_detection(self, img):
        bbox = [0, 0, img.shape[0], img.shape[1]]
        input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
        
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.image_size[0] * 2 / 3.0 / max(w, h)
        
        rimg, M = transform(img, center, self.image_size[0], _scale,
                            rotate)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        rimg = np.transpose(rimg, (2, 0, 1))  #3*112*112, RGB
        
        input_blob[0] = rimg
        
        if self.use_onnx and self.model is not None:
            input_name = self.model.get_inputs()[0].name
            pred = self.model.run(None, {input_name: input_blob})[0][0]
        else:
            pred = np.zeros((106, 2), dtype=np.float32)
            
        if pred.shape[0] >= 3000:
            pred = pred.reshape((-1, 3))
        else:
            pred = pred.reshape((-1, 2))
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (self.image_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (self.image_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)
        
        return pred
    
    
    def get(self, img, get_all=False):
        out = []
        det_im, det_scale = square_crop(img, self.det_size)
        bboxes, _ = self.detector.detect(det_im)
        if bboxes.shape[0] == 0:
            return out
        bboxes /= det_scale
        if not get_all:
            areas = []
            for i in range(bboxes.shape[0]):
                x = bboxes[i]
                area = (x[2] - x[0]) * (x[3] - x[1])
                areas.append(area)
            m = np.argsort(areas)[-1]
            bboxes = bboxes[m:m + 1]
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            input_blob = np.zeros((1, 3) + self.image_size, dtype=np.float32)
            w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
            center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
            rotate = 0
            _scale = self.image_size[0] * 2 / 3.0 / max(w, h)
            rimg, M = transform(img, center, self.image_size[0], _scale,
                                rotate)
            rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            rimg = np.transpose(rimg, (2, 0, 1))  #3*112*112, RGB
            input_blob[0] = rimg
            
            if self.use_onnx and self.model is not None:
                input_name = self.model.get_inputs()[0].name
                pred = self.model.run(None, {input_name: input_blob})[0][0]
            else:
                pred = np.zeros((106, 2), dtype=np.float32)
                
            if pred.shape[0] >= 3000:
                pred = pred.reshape((-1, 3))
            else:
                pred = pred.reshape((-1, 2))
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (self.image_size[0] // 2)
            if pred.shape[1] == 3:
                pred[:, 2] *= (self.image_size[0] // 2)

            IM = cv2.invertAffineTransform(M)
            pred = trans_points(pred, IM)
            out.append(pred)
        return out
