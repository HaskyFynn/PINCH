import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional

# Mocking necessary classes
@dataclass
class MarkerProfile:
    marker_id: str
    enroll_embs: Optional[List[List[float]]] = None

@dataclass
class Registry:
    markers: List[MarkerProfile]

EMBED_DIM = 128
THRESH_PERCENTILE = 5
DEVICE = "cpu"

def build_session_classifier(registry: Registry) -> Optional[Dict[str, np.ndarray]]:
    if registry is None:
        return None

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    names: List[str] = []

    print(f"Registry has {len(registry.markers)} markers.")

    for idx, m in enumerate(registry.markers):
        embs = getattr(m, "enroll_embs", None)
        if embs is None:
            print(f"Skipping marker {idx} ({m.marker_id}) due to missing embeddings.")
            continue
        
        arr = np.asarray(embs, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[0] == 0 or arr.shape[1] != EMBED_DIM:
            print(f"Skipping marker {idx} ({m.marker_id}) due to invalid shape.")
            continue
            
        print(f"Adding marker {idx} ({m.marker_id}) with label {idx}.")
        X_list.append(arr)
        y_list.append(np.full((arr.shape[0],), idx, dtype=np.int64))
        names.append(m.marker_id)

    if not X_list or len(names) < 2:
        print("Not enough data for classifier.")
        return None

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    device = DEVICE
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)

    n_classes = len(names)
    print(f"n_classes (len(names)): {n_classes}")
    print(f"Unique labels in y: {np.unique(y)}")
    
    model = nn.Linear(EMBED_DIM, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)

    print("Starting training loop...")
    try:
        logits = model(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        print("Training step successful.")
    except IndexError as e:
        print(f"CAUGHT EXPECTED ERROR: {e}")
        return "ERROR_CAUGHT"
    except RuntimeError as e:
        print(f"CAUGHT EXPECTED ERROR: {e}")
        return "ERROR_CAUGHT"
    
    return "SUCCESS"

def test_reproduction():
    # Create 3 markers. Middle one has no embeddings.
    m1 = MarkerProfile("marker1", [[0.1]*EMBED_DIM for _ in range(10)])
    m2 = MarkerProfile("marker2", None) # Missing embeddings -> Skip
    m3 = MarkerProfile("marker3", [[0.2]*EMBED_DIM for _ in range(10)])
    
    registry = Registry(markers=[m1, m2, m3])
    
    result = build_session_classifier(registry)
    
    if result == "ERROR_CAUGHT":
        print("\n>>> REPRODUCTION SUCCESSFUL: The classifier crashed due to index gaps.")
    else:
        print("\n>>> REPRODUCTION FAILED: The classifier did not crash.")

if __name__ == "__main__":
    test_reproduction()
