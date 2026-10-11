"""Check SCRFD coordinate decoding and the landmark contract with AdaFace."""
import unittest
import numpy as np
from test.web_remote.scrfd_adapter import WebSCRFDDetector
from test.web_remote.adaface_engine import AdaEngine
from app.recognition.adaface_embedder import AdaFaceEmbedder

class SCRFDTests(unittest.TestCase):
    def detector(self):
        detector = WebSCRFDDetector.__new__(WebSCRFDDetector)
        detector.input_size=(640,640)
        detector.score_threshold=.5
        detector.nms_threshold=.4
        detector.centers={}
        return detector
    def outputs(self):
        sizes=(12800,3200,800)
        return ([np.zeros((n,1),np.float32) for n in sizes]+
                [np.zeros((n,4),np.float32) for n in sizes]+
                [np.zeros((n,10),np.float32) for n in sizes])
    def test_rgb_preprocessing_keeps_source_unchanged(self):
        frame=np.zeros((360,1280,3),np.uint8)
        frame[:]=[10,20,30]
        blob,scale=self.detector()._preprocess(frame)
        self.assertEqual(blob.shape,(1,3,640,640))
        self.assertEqual(scale,.5)
        self.assertTrue(np.allclose(blob[0,:,0,0],(np.array([30,20,10])-127.5)/128))
        self.assertTrue(np.all(frame==[10,20,30]))
    def test_decode_scale_nms_and_landmark_order(self):
        detector=self.detector()
        outputs=self.outputs()
        at=(20*80+20)*2
        offsets=np.array([[-2,-1],[2,-1],[0,0],[-1,2],[1,2]],np.float32)
        for index,score in ((at,.9),(at+1,.8)):
            outputs[0][index]=score
            outputs[3][index]=[4,4,4,4]
            outputs[6][index]=offsets.reshape(-1)
        faces=detector._decode(outputs,(720,1280,3),.5)
        self.assertEqual(len(faces),1)
        face=faces[0]
        self.assertEqual((face.x,face.y,face.width,face.height),(256,256,128,128))
        expected=(np.array([160,160])+offsets*8)/.5
        self.assertTrue(np.allclose(face.landmarks,expected[[1,0,2,4,3]]))
        # Core AdaFace reverses the YuNet convention: recover native SCRFD order.
        self.assertTrue(np.allclose(face.landmarks[[1,0,2,4,3]],expected))
    def test_no_detections_and_padding_rejected(self):
        detector=self.detector()
        outputs=self.outputs()
        self.assertEqual(detector._decode(outputs,(720,1280,3),.5),[])
        at=(70*80+20)*2
        outputs[0][at]=.9
        outputs[3][at]=[1,1,1,1]
        self.assertEqual(detector._decode(outputs,(720,1280,3),.5),[])
    def test_ada_alignment_and_preprocessing_are_reused(self):
        self.assertIs(AdaEngine._align,AdaFaceEmbedder._align)
        self.assertIs(AdaEngine._preprocess,AdaFaceEmbedder._preprocess)

if __name__=='__main__': unittest.main()
