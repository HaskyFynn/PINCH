from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import tempfile
import unittest
import numpy as np
from pinch.config import Settings
from pinch.registry import Registry, MarkerProfile
from pinch.pipeline import Pipeline, Observation
from pinch.evaluation import Evaluator
from pinch.worker import Worker
from pinch.enrollment import Enrollment
from pinch.source import Frame
from unittest.mock import Mock


def v(i):
    x=np.zeros(128,dtype=np.float32)
    x[i]=1
    return x


class Boxes:
    """Minimal detection result using the installed ByteTrack input protocol."""
    def __init__(self,rows):
        self.data=np.array(rows,dtype=np.float32).reshape(-1,6)
        self.xyxy=self.data[:,:4]
        self.conf=self.data[:,4]
        self.cls=self.data[:,5]
        self.xywh=self.xyxy.copy()
        self.xywh[:,:2]=(self.xyxy[:,:2]+self.xyxy[:,2:])/2
        self.xywh[:,2:]=self.xyxy[:,2:]-self.xyxy[:,:2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return Boxes(self.data[index])


class FakeModels:
    embedder_hash='test'
    def __init__(self,rows):
        self.rows=rows
        self.embedding_calls=0

    def detect(self,frame):
        return Boxes(self.rows)

    def embed(self,crops):
        self.embedding_calls+=1
        result=[]
        for crop in crops:
            blue=np.median(crop[:,:,2])
            result.append(v(int(round((blue-30)/50))))
        return np.array(result,dtype=np.float32).reshape(-1,128)


def scene(count=4):
    frame=np.zeros((180,560,3),dtype=np.uint8)
    rows=[]
    for i in range(count):
        x=20+i*130
        rows.append([x,40,x+80,120,.95,0])
        frame[40:120,x:x+80,0]=30+50*i
        frame[40:120:2,x:x+80,1]=255
    return frame,rows


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The real ByteTrack implementation is used; only detections/features are fixtures.
        import os
        Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True,exist_ok=True)

    def setUp(self):
        self.frame,rows=scene()
        self.models=FakeModels(rows)
        self.registry=Registry([MarkerProfile(chr(65+i),[v(i).tolist()],.8) for i in range(4)])
        self.pipeline=Pipeline(self.models,self.registry,Settings())

    def warm(self):
        for i in range(3):
            result,metrics=self.pipeline.process(self.frame,i*.1)
        return result,metrics

    def test_real_bytetrack_four_markers_and_batched_embeddings(self):
        result,metrics=self.warm()
        self.assertEqual({o.name for o in result},{'A','B','C','D'})
        self.assertEqual(metrics['raw_detections'],4)
        self.assertEqual(metrics['identified'],4)
        self.assertEqual(self.models.embedding_calls,3)

    def test_four_identities_can_confirm_at_slow_processing_rates(self):
        for now in (0.,2.,4.):
            result,_=self.pipeline.process(self.frame,now)
        self.assertEqual({o.name for o in result},{'A','B','C','D'})

    def test_low_confidence_detections_recover_existing_tracks(self):
        result,_=self.warm()
        ids={o.track_id for o in result}
        for row in self.models.rows:
            row[4]=.15
        result,metrics=self.pipeline.process(self.frame,.3)
        self.assertEqual({o.track_id for o in result},ids)
        self.assertEqual(metrics['tracked_detections'],4)

    def test_mixed_confidence_stages_keep_exact_crop_owners(self):
        self.warm()
        self.models.rows[0][4] = .15
        self.models.rows[2][4] = .2
        result,metrics = self.pipeline.process(self.frame,.3)
        self.assertEqual(metrics['usable_crops'],4)
        self.assertEqual({o.name for o in result},{'A','B','C','D'})

    def test_blur_does_not_remove_detected_rois(self):
        self.warm()
        result,metrics=self.pipeline.process(np.zeros_like(self.frame),.3)
        self.assertEqual(len(result),4)
        self.assertTrue(all(o.detected and o.quality=='blurred' for o in result))
        self.assertEqual(metrics['usable_crops'],0)
        self.assertEqual(metrics['identified'],4)

    def test_missing_boxes_predicted_briefly_and_then_removed(self):
        self.warm()
        self.models.rows=[]
        result,metrics=self.pipeline.process(self.frame,.3)
        self.assertEqual(len(result),4)
        self.assertTrue(all(not o.detected for o in result))
        self.assertEqual(metrics['identified'],0)
        result,_=self.pipeline.process(self.frame,.7)
        self.assertEqual(result,[])

    def test_reset_discards_old_registry_identities(self):
        self.warm()
        self.pipeline.reset(Registry())
        result,_=self.pipeline.process(self.frame,.3)
        self.assertTrue(all(o.name=='unknown' for o in result))

    def test_overlap_rejects_appearance_without_hiding_boxes(self):
        self.models.rows=[[20,40,100,120,.95,0],[40,40,120,120,.95,0]]
        result,metrics=self.pipeline.process(self.frame,0)
        self.assertEqual(metrics['usable_crops'],0)
        self.assertTrue(any(o.quality=='overlapping' for o in result))

    def test_smoothed_roi_is_not_mistaken_for_a_second_marker(self):
        self.frame,self.models.rows=scene(1)
        tracker=SimpleNamespace(update=lambda boxes,frame:np.array([[25,40,105,120,1,.95,0,0]]), reset=lambda:None)
        pipeline=Pipeline(self.models,self.registry,Settings(),tracker=tracker)
        result,metrics=pipeline.process(self.frame,0)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].quality,'usable')
        self.assertEqual(metrics['usable_crops'],1)

    def test_enrollment_does_not_choose_highest_of_several_markers(self):
        worker=Worker(Settings())
        worker.models=self.models
        worker.enrollment=Enrollment('new')
        worker.enroll_frame(Frame(self.frame,0,0,0))
        self.assertEqual(worker.enrollment.count,0)

    def test_enrollment_pauses_for_a_weak_second_marker(self):
        self.models.rows = self.models.rows[:2]
        self.models.rows[1][4] = .15
        worker = Worker(Settings())
        worker.models = self.models
        worker.enrollment = Enrollment('new')
        worker.enroll_frame(Frame(self.frame,0,0,0))
        self.assertEqual(worker.enrollment.count,0)


class EvaluationTests(unittest.TestCase):
    def test_misses_unknowns_wrong_names_and_unenrolled_are_separate(self):
        e=Evaluator(['A','B','C','D'])
        gt=[{'marker_id':chr(65+i),'box':[i*20,0,i*20+10,10]} for i in range(4)]
        predictions=[Observation(1,gt[0]['box'],.9,name='A'),
                     Observation(2,gt[1]['box'],.9,name='unknown'),
                     Observation(3,gt[2]['box'],.9,name='D')]
        e.add(gt,predictions)
        s=e.summary()
        self.assertEqual(s['detection_recall_iou_05'],.75)
        self.assertEqual(s['correct_identity_given_detection'],1/3)
        self.assertEqual(s['wrong_name_given_detection'],1/3)
        self.assertEqual(s['false_unknown_given_detection'],1/3)
        self.assertEqual(s['all_four_correct_frame_rate'],0)

    def test_predicted_roi_does_not_count_as_a_detection(self):
        e=Evaluator(['A'])
        e.add([{'marker_id':'A','box':[0,0,10,10]}],[Observation(1,[0,0,10,10],.9,detected=False,name='A')])
        self.assertEqual(e.summary()['detection_recall_iou_05'],0)


class WorkerCommandTests(unittest.TestCase):
    def test_bad_registry_import_does_not_stop_the_running_session(self):
        worker=Worker(Settings())
        worker.mode='run'
        worker.session=Mock()
        with self.assertRaises(FileNotFoundError):
            worker.handle('import',{'path':'does-not-exist-registry.json'})
        self.assertEqual(worker.mode,'run')
        worker.session.close.assert_not_called()

    def test_stopping_session_leaves_a_valid_preview_state(self):
        worker=Worker(Settings())
        worker.mode='run'
        worker.session=Mock()
        worker.stop_session()
        self.assertEqual(worker.mode,'preview')
        self.assertIsNone(worker.session)


if __name__=='__main__':
    unittest.main()
