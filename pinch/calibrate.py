"""Propose thresholds from separately annotated replay sessions.

Writes a candidate registry, never silently modifies the active registry.
Use a different recording to measure the candidate's generalization.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from .registry import Registry
from .evaluation import iou


def choose_threshold(positive, negative, max_false_accept=.01):
    positive,negative=np.asarray(positive,dtype=float),np.asarray(negative,dtype=float)
    if len(positive)<20 or len(negative)<20:
        return None
    # Lowest threshold that meets the empirical negative acceptance bound.
    candidates=np.unique(np.r_[positive,negative,np.nextafter(negative,np.inf)])
    candidates=candidates[(candidates>=-1)&(candidates<=1)]
    allowed=[t for t in candidates if np.mean(negative>=t)<=max_false_accept]
    if not allowed:
        return None
    threshold=float(min(allowed))
    recall=float(np.mean(positive>=threshold))
    if recall==0:
        return None
    return {'threshold':threshold, 'positive_acceptance':recall,
            'negative_acceptance':float(np.mean(negative>=threshold)),
            'positive_samples':len(positive), 'negative_samples':len(negative)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--output',required=True,help='New candidate registry path')
    parser.add_argument('--max-false-accept',type=float,default=.01)
    args=parser.parse_args()
    if not 0<=args.max_false_accept<1:
        raise ValueError('max-false-accept must be in [0,1)')
    directory=Path(args.session)
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    registry=Registry.from_json(metadata['registry'])
    annotated={int(f['frame']):f['markers'] for f in json.loads(Path(args.annotations).read_text())['frames']}
    samples={m.marker_id:{'positive':[],'negative':[]} for m in registry.markers}
    with (directory/'frames.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line)
            index=row.get('recorded_frame')
            if index is None:
                index=row['frame']
            gt=[g for g in annotated.get(index,[]) if g.get('visible',True)]
            pred=[o for o in row['observations'] if o['detected'] and o.get('scores')]
            if not gt or not pred:
                continue
            matrix=np.array([[iou(g['box'],p['box']) for p in pred] for g in gt])
            rr,cc=linear_sum_assignment(np.where(matrix>=.5,1-matrix,2.))
            for r,c in zip(rr,cc):
                if matrix[r,c]<.5:
                    continue
                for name,score in pred[c]['scores'].items():
                    if name in samples:
                        kind='positive' if gt[r]['marker_id']==name else 'negative'
                        samples[name][kind].append(score)
    reports={}
    for profile in registry.markers:
        data=samples[profile.marker_id]
        result=choose_threshold(data['positive'],data['negative'],args.max_false_accept)
        reports[profile.marker_id]=result or {'unchanged':True,'reason':'Need at least 20 positive and 20 negative usable samples with separable scores.'}
        if result:
            profile.thr=result['threshold']
            profile.calibration='annotated_recording_empirical_threshold'
    output=Path(args.output).resolve()
    configured=Path(metadata['settings']['registry'])
    from .config import ROOT
    active=configured if configured.is_absolute() else ROOT/configured
    if output==active.resolve():
        raise ValueError('Choose a separate candidate registry path, then evaluate it before importing')
    registry.save(output)
    report={'markers':reports,'note':'Empirical calibration on this recording only. Correlated frames are not independent trials. Test on a separate recording before importing.'}
    Path(str(output)+'.report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
