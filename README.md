# PINCH — multi-marker desktop reader

Enroll physical tags, track their regions, and identify multiple tags in one camera frame. The runtime supports at least four concurrent tracks; the default detection limit is 32. Accuracy still depends on tag appearance, the trained models, enrollment coverage, lighting, and camera performance.

## Start

On this PC, double-click **Start PINCH.cmd**. It uses a project `.venv` when present, otherwise the existing `Desktop\Experiment\.venv`. The camera stays closed until you click **Open camera**.

Both `python pinchreader.py` and `python letsgo/pinchreader.py` launch the same app. For a fresh environment, use Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-live.txt
.\.venv\Scripts\python -m pinch.app
```

Settings are in `config/live.json`; paths resolve against this project. Existing detector and embedding weights are in `letsgo/`. Loading does not download ImageNet weights. CPU is the default, matching the environment verified on this PC.

## Enroll and test live

1. Open camera 0, or another camera index. Keep tag patterns large and clear in the frame.
2. This local copy includes four existing profiles in `data/registry.json`. Legacy profiles work without optional density statistics; original thresholds are preserved. Personal profiles are not published to GitHub.
3. If an old tag consistently reports `below threshold`, re-enroll it under the same name using the current camera and model.
4. Enroll each tag **separately**, keeping other tags outside the frame. Give each physical tag its own name, even if several belong to one person.
5. Follow five view prompts. Collect eight clear samples per view, then click **Next view**. Elapsed time alone does not complete enrollment. Finish with **Save enrollment**. Replacement and similar-looking profiles are shown for review before saving; a registry backup is kept.
6. Start recognition with two tags, then three, then four. Test movement, crossings, partial overlap, distance changes, and exit/re-entry. Recognition continues until stopped.
7. To retain a reproducible record, check **Save video with session logs** before starting. Recordings stay in `runs/live/`. They contain processed raw frames, not every camera frame; a sidecar retains original timestamps and logs count skipped camera frames.

Track wrong names separately from false unknowns. A wrong name is not an improvement over `unknown`.

## Display and behavior

| Display | Meaning |
| --- | --- |
| Solid ROI | Current measured detection, tracked or pending confirmation. |
| Dashed ROI | Brief predicted position after detection loss; expires after 0.4 seconds by default. |
| `blurred`, `too small`, `overlapping` | ROI remains visible; appearance does not update identity. |
| `confirming` | Candidate needs repeated evidence. |
| `below threshold` | Best similarity does not meet the saved threshold. |
| `ambiguous` | Best and second-best identities are too close. |
| `identity conflict` | Several tracks claim one physical marker. |
| `held ...` | A name is briefly retained; expires after 0.6 seconds without accepted evidence. |
| `model mismatch` | Profile came from another embedding checkpoint; re-enroll it. |

The table shows the best candidate and similarity versus required threshold. Cosine scores are not probabilities. Read-to-display latency excludes sensor exposure and buffering inside the camera driver.

### Design choices and tradeoffs

- Detection, tracking, crop quality, and identity are separate. A poor crop no longer removes a detected ROI.
- ByteTrack receives weak detections for recovery; stricter gates control new tracks and identities. Lower detection confidence may introduce more false detections.
- Embeddings use the exact measured detection assigned by ByteTrack, not a lagging motion box. Crops are embedded together and scored against cached prototype arrays.
- Names require absolute similarity and ambiguity checks. Conflicts are resolved across tracks; losing tracks are not forced into second choices. Difficult crossings may briefly show unknown instead of a wrong name.
- Retained names and ROIs expire by elapsed time. Longer holds can preserve stale names or phantom boxes. Confirmation uses a separate gap limit so slower inference can still accumulate consistent evidence.
- Historical maximum confidence and classifier rescue were removed. New evidence can correct old names; no classifier label gaps or mixed probability/cosine comparisons remain.
- Optional density rejection is disabled by default pending calibration. Missing density statistics never invalidate legacy profiles.
- Enrollment reserves a temporal block from each view for threshold estimation. This is within-session calibration, not proof of generalization to another recording.
- GUI, capture, and inference are separate. A bounded latest-frame buffer limits stale frames but cannot make slow inference process every camera frame.
- Registry saves validate first, replace atomically, and retain a backup. Reload/import resets identity state.

## Replay and evaluation

Run from the project folder with its Python environment:

```powershell
python -m pinch.replay --video "C:\recordings\four-tags.mp4"
python -m pinch.replay --video "C:\recordings\four-tags.mp4" --annotations "C:\recordings\labels.json"
```

No GUI or camera is required. `--max-frames 100` limits a smoke test. Keep PINCH's `capture.avi` and `video_timestamps.jsonl` together to restore original timing.

Annotations use zero-based video frames and original image pixel coordinates:

```json
{"frames": [{"frame": 0, "markers": [
  {"marker_id": "Alice", "box": [100, 120, 180, 200], "visible": true},
  {"marker_id": "Bob", "box": [300, 120, 380, 200], "visible": true}
]}]}
```

Annotate independently of predictions. Include unenrolled distractors under distinct names. `visible: false` excludes unobservable markers; omitted frames are not evaluated. Annotate densely around crossings and re-entry.

Evaluation uses IoU >= 0.5 and reports detection recall, false detections, correct/wrong/unknown identity rates, unenrolled rejection, all-four correctness, and tracker ID changes. Predicted ROIs never count as detections. Without annotations, accuracy remains null.

## Calibrate thresholds

```powershell
python -m pinch.calibrate --session "runs\live\SESSION" --annotations "C:\recordings\labels.json" --output "data\candidate-registry.json"
python -m pinch.replay --video "C:\recordings\different-session.mp4" --registry "data\candidate-registry.json" --annotations "C:\recordings\different-session.json"
```

For PINCH recordings, annotate `capture.avi` frames; calibration maps them to logged scores. Each profile needs at least 20 positive and 20 negative usable samples. The default empirical false-accept ceiling is 1%. Adjacent frames are correlated, so this is not a statistical guarantee. Inseparable profiles remain unchanged.

Calibration writes a candidate registry and report, refusing to overwrite the active registry. Evaluate on another recording, then **Import registry** if the tradeoff is acceptable. Do not lower all thresholds simply to remove unknown labels.

## Modules, logs and tests

| Module in `pinch/` | Responsibility |
| --- | --- |
| `config.py`, `registry.py` | Settings, validation and safe persistence. |
| `enrollment.py` | Guided samples, prototypes and initial thresholds. |
| `vision.py`, `pipeline.py` | Models, ByteTrack, ROIs and crop quality. |
| `identity.py` | Batched scoring, ambiguity/conflict checks and temporal state. |
| `source.py`, `worker.py`, `app.py` | Capture, inference ownership and GUI. |
| `session.py` | Incremental logs, optional recording and provenance. |
| `replay.py`, `evaluation.py`, `calibrate.py` | Repeatable evaluation and calibration. |

Logs include model hashes, settings, registry snapshot, versions, raw boxes, crop reasons, IDs, scores, thresholds and stage timings. Personal registries and new recordings are ignored by Git.

```powershell
python -m unittest discover -s tests -v
```

The base is `Desktop\Experiment` at `c845e52`, matching GitHub `main` when inspected, plus its uncommitted GUI edits from 18 February 2026. The exact edited source is in `legacy/pinchreader_20260218.py`, SHA-256 `f823a42efe40e49b3ec32fabbf6203b24a66c61499218a18f694db6a216dd68b`. Reproduction scripts are preserved too. The newer training notebook remains local and is not part of this push or the live runtime. Old experiment requirements are in `legacy/requirements-experiments.txt`.

Desktop copies were not modified. This working copy uses branch `improve-four-marker-reliability`. Use current launchers; `legacy/` is archival reference. See `IMPLEMENTATION-REPORT.md` for verification and remaining physical live-test requirements.
