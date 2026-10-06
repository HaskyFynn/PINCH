"""Replay a recording without opening a camera or requiring the desktop GUI."""
import argparse
import json
from pathlib import Path
from .config import load_settings
from .registry import Registry
from .vision import Models
from .pipeline import Pipeline
from .source import VideoSource
from .session import Session
from .evaluation import Evaluator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',required=True)
    parser.add_argument('--config')
    parser.add_argument('--registry')
    parser.add_argument('--annotations',help='JSON with frames: [{frame, markers: [{marker_id, box, visible}]}]')
    parser.add_argument('--max-frames',type=int,default=0)
    args=parser.parse_args()
    settings=load_settings(args.config)
    registry=Registry.load(args.registry or settings.path('registry'))
    annotations={}
    if args.annotations:
        data=json.loads(Path(args.annotations).read_text(encoding='utf-8'))
        for frame in data['frames']:
            index=int(frame['frame'])
            if index in annotations:
                raise ValueError(f'Duplicate annotated frame {index}')
            for marker in frame['markers']:
                box=marker['box']
                if len(box)!=4 or not box[0]<box[2] or not box[1]<box[3]:
                    raise ValueError(f'Invalid annotation box in frame {index}')
            annotations[index]=frame['markers']
    models=Models(settings)
    pipeline=Pipeline(models,registry,settings)
    source=VideoSource(args.video,paced=False)
    session=Session(settings,models,registry,args.video)
    evaluator=Evaluator(registry.names())
    try:
        while True:
            frame=source.read()
            if frame is None:
                if source.error:
                    raise RuntimeError(source.error)
                break
            observations,metrics=pipeline.process(frame.image,frame.timestamp)
            session.write(frame,observations,metrics)
            if frame.index in annotations:
                evaluator.add(annotations[frame.index],observations)
            if args.max_frames and session.count>=args.max_frames:
                break
    finally:
        source.close()
        session.close()
    report=evaluator.summary() if args.annotations else {'identity_accuracy':None,'note':'No annotations supplied; this run measures processing behavior, not recognition accuracy.'}
    (session.directory/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(session.directory),'frames':session.count,'evaluation':report},indent=2))


if __name__=='__main__':
    main()
