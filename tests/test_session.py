from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from pinch.config import Settings
from pinch.registry import Registry
from pinch.source import Frame,VideoSource
from pinch.session import Session


class SessionTests(unittest.TestCase):
    def test_recording_roundtrip_preserves_timing_and_frame_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=replace(Settings(),runs=tmp)
            models=SimpleNamespace(detector_hash='det',embedder_hash='emb',device='cpu')
            session=Session(settings,models,Registry(),'test camera',record_video=True)
            image=np.zeros((64,64,3),dtype=np.uint8)
            for i,stamp in enumerate((10.,10.4,10.9)):
                session.write(Frame(image,10+i*4,stamp,stamp),[],{'processing_ms':5.})
            session.close()
            source=VideoSource(session.directory/'capture.avi',paced=False)
            frames=[source.read() for _ in range(3)]
            self.assertEqual([f.timestamp for f in frames],[10.,10.4,10.9])
            self.assertIsNone(source.read())
            source.close()
            import json
            rows=[json.loads(x) for x in (session.directory/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['recorded_frame'] for r in rows],[0,1,2])
            self.assertEqual([r['frame'] for r in rows],[10,14,18])
            self.assertIsNone(json.loads((session.directory/'summary.json').read_text())['identity_accuracy'])
