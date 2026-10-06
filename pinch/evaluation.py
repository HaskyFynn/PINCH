"""Evaluate physical marker annotations independently of tracker IDs."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda x:max(0,x[2]-x[0])*max(0,x[3]-x[1])
    return inter/max(area(a)+area(b)-inter,1e-9)


class Evaluator:
    def __init__(self, registered_names):
        self.names=set(registered_names)
        self.visible=self.matched=self.correct=self.wrong=self.unknown=0
        self.enrolled_matched=self.enrolled_visible=0
        self.unenrolled_matched=self.unenrolled_rejected=0
        self.false_detections=self.frames=self.all_four_frames=self.all_four_correct=0
        self.switches=0
        self.previous_tracks={}
        self.ious=[]

    def add(self,annotations,observations):
        gt=[g for g in annotations if g.get('visible',True)]
        predictions=[o for o in observations if o.detected]
        self.frames+=1
        self.visible+=len(gt)
        self.enrolled_visible+=sum(g['marker_id'] in self.names for g in gt)
        frame_correct=0
        pairs=[]
        if gt and predictions:
            matrix=np.array([[iou(g['box'],p.box) for p in predictions] for g in gt])
            rows,cols=linear_sum_assignment(np.where(matrix>=.5,1-matrix,2.))
            pairs=[(r,c) for r,c in zip(rows,cols) if matrix[r,c]>=.5]
        self.matched+=len(pairs)
        self.false_detections+=len(predictions)-len(pairs)
        for r,c in pairs:
            g,p=gt[r],predictions[c]
            self.ious.append(iou(g['box'],p.box))
            name=g['marker_id']
            if name in self.names:
                self.enrolled_matched+=1
                if p.name==name:
                    self.correct+=1
                    frame_correct+=1
                elif p.name=='unknown':
                    self.unknown+=1
                else:
                    self.wrong+=1
            else:
                self.unenrolled_matched+=1
                self.unenrolled_rejected+=p.name=='unknown'
            if p.track_id>=0:
                if name in self.previous_tracks and self.previous_tracks[name]!=p.track_id:
                    self.switches+=1
                self.previous_tracks[name]=p.track_id
        eligible=sum(g['marker_id'] in self.names for g in gt)
        if eligible>=4:
            self.all_four_frames+=1
            self.all_four_correct+=frame_correct==eligible

    def summary(self):
        ratio=lambda n,d: n/d if d else None
        return {'annotated_frames':self.frames, 'visible_markers':self.visible,
                'detection_recall_iou_05':ratio(self.matched,self.visible),
                'false_detections_per_annotated_frame':ratio(self.false_detections,self.frames),
                'correct_identity_given_detection':ratio(self.correct,self.enrolled_matched),
                'correct_identity_over_visible_enrolled':ratio(self.correct,self.enrolled_visible),
                'wrong_name_given_detection':ratio(self.wrong,self.enrolled_matched),
                'false_unknown_given_detection':ratio(self.unknown,self.enrolled_matched),
                'unenrolled_rejection_given_detection':ratio(self.unenrolled_rejected,self.unenrolled_matched),
                'all_four_correct_frame_rate':ratio(self.all_four_correct,self.all_four_frames),
                'all_four_eligible_frames':self.all_four_frames,
                'tracker_id_changes_on_annotated_physical_markers':self.switches,
                'mean_matched_iou':float(np.mean(self.ious)) if self.ious else None}
