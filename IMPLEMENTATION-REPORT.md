# PINCH implementation and verification

Updated 1 October 2026.

## Delivered

The current desktop reader has been refactored into separate enrollment, model,
tracking, identity, capture, persistence, GUI, and evaluation modules. Both original
entry points launch the same application. `Start PINCH.cmd` uses the existing Python
environment on this PC; the camera opens only when requested in the GUI.

The baseline is GitHub main at `c845e52`, plus the more recent uncommitted desktop
GUI from 18 February 2026. That complete source is preserved byte-for-byte in
`legacy/pinchreader_20260218.py`. The desktop originals were left intact. Work is on
`improve-four-marker-reliability`.

### Changes that address the reported failures

| Failure or risk | Change | Tradeoff |
| --- | --- | --- |
| Legacy prototypes rejected when optional density data was missing | Prototype matching works without density fields; malformed statistics give a validation error. | Legacy thresholds can still be too strict for a different camera. |
| Classifier label gaps and incompatible confidence scales | Removed classifier rescue; one normalized cosine matching path. | No alternate classifier rescues a weak prototype match. |
| Old maximum confidence prevented identity correction | Evaluate current evidence and confirm consistent candidates. | Three observations delay the first name. |
| Names persisted indefinitely or moved between tags | Bounded holds, explicit conflicts, ambiguity rejection, and current appearance evidence. | Crossings can temporarily become unknown. |
| A weak detector result disappeared immediately | ByteTrack uses lower-confidence detections to recover established tracks. | Weak false detections may remain visible. |
| Tracker smoothing could crop the wrong pattern | Use ByteTrack's exact matched raw detection for embeddings. | This adapter requires the pinned Ultralytics version. |
| Blur or partial overlap hid the ROI | Keep the tracked ROI while rejecting unreliable appearance updates. | A visible ROI does not imply a reliable name. |
| Enrollment collected the wrong tag or incomplete samples | Five required views; one confident marker and no second plausible detection; validate before saving. | Weak false detections may pause enrollment. |
| More tags multiplied embedding calls | Batch all usable crops and cache the prototype matrix. | CPU inference remains a throughput limit. |
| A busy inference loop froze controls or displayed a backlog | Separate GUI, capture, and inference, with bounded frame queues. | Skipped camera frames cannot be recovered by tracking. |

Names are unique per physical tag. Several tags belonging to one person need
different marker names. Existing profiles are available locally; private profiles
and new recordings are excluded from Git.

## Verification evidence

- **46 regression checks:** registry compatibility and atomic saves; enrollment;
  four simultaneous identities; conflicts, crossings, expiry and re-entry; real
  ByteTrack with controlled detector/features; mixed high/low-confidence crop
  ownership; GUI construction and commands with a simulated worker; replay
  evaluation; threshold calibration; recording/timestamp roundtrip.
- **Actual model replay:** the final recognition pipeline processed 12 frames from
  the available local `Video1.mp4` with the existing YOLO and ResNet checkpoints.
  Median processing time was 276 ms and the 95th percentile was 444 ms. This short,
  unlabeled clip is a smoke test, not an accuracy benchmark.
- **Synthetic load check:** one, two, three, and four repeated copies of one frame
  yielded the corresponding number of detected, tracked, and usable regions.
  Four regions took a median 1,985 ms and a 95th percentile 2,280 ms in this run.
  Every copy was the same physical pattern; this does not establish four distinct
  identities. All reported names were unknown.
- Evidence is in `validation/`. GitHub Actions is configured to run the regression
  suite on Windows with Python 3.13; local results do not establish a CI result.

Tests ran using Python 3.13.14, PyTorch 2.10.0 CPU, and Ultralytics 8.4.10. This PC
has an Intel i5-10310U and Intel UHD graphics, with no CUDA device available.
Timing varies with desktop load; the short replay and synthetic scenes are
different workloads and should not be compared as a speedup measurement.

Model hashes:

- YOLO: `c49c954ab783062827573f8123969b1911f7f96a6c7a3225c3316a033e457f83`
- Embedder: `14d2305fcfd51b7088a4a24bf987bc0931c6d4046db4146694027d51f5bbb574`

## Limits and physical acceptance test

**Smooth four-tag live tracking and real-world identification accuracy have not
been demonstrated.** No physical camera test with four distinct tags was available.
The synthetic CPU result is too slow for fast interaction. The app now warns about
slow processing and preserves enough evidence to measure it. Longer confirmation
gaps allow consistent slow frames to confirm, but do not fix missed motion.
Manual visual GUI inspection was unavailable because computer-use approval timed
out; automated Tk checks passed.

Use the README enrollment and recording steps to run the following acceptance
test with the actual camera and printed tags:

1. Enroll or refresh each of four tags separately under its own name. Include
   realistic angles, distances, and lighting. Review any similarity warning.
2. Record stationary tags, then add movement with two, three, and four visible.
   Include crossings, overlap, a brief obstruction, exit/re-entry, and one
   unenrolled distractor. Repeat under different lighting.
3. Annotate the saved recording independently. Evaluate detection recall, wrong
   names, false unknowns, all-four correctness, track changes, and processing time.
   A lower unknown rate is acceptable only if it does not hide more wrong names.
4. For threshold failures, calibrate a separate candidate registry and evaluate
   it on another recording before importing it. For detector misses, collect the
   missed scenes for detector evaluation/training; thresholds cannot invent a ROI.
5. If the CPU still cannot keep up, evaluate smaller detector inputs against
   small-tag recall, or an optimized inference backend/appropriate GPU. Do not
   lower embedding resolution or quantize the trained model without comparing
   accuracy and recalibrating profiles.

The code and repeatable test workflow are delivered. The physical acceptance test
remains necessary before describing the system as reliable for live interaction.
