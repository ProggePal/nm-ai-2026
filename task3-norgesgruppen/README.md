# NorgesGruppen Product Detection

Two-stage computer vision pipeline for retail shelf product detection and classification. Built for NM i AI 2026, Task 3.

## Architecture

**Stage 1 — Detection (70% of score)**
YOLOv11x fine-tuned on the NorgesGruppen COCO dataset for category-agnostic bounding box detection.

**Stage 2 — Classification (30% of score)**
ResNet50 feature extractor with cosine similarity matching against a pre-built reference feature bank (`feature_bank.json`). Product crops from Stage 1 are matched to the closest product in the bank.

## Submission

Build the zip:

```bash
./build_submission.sh
```

Produces a zip containing `run.py`, model weights, and the feature bank — ready to upload at [app.ainm.no](https://app.ainm.no).

Expects `best.onnx` (preferred) or `best.pt` in the working directory. `run.py` auto-detects which is present.

## Local testing

Mirrors the competition sandbox (8 GB RAM, 4 vCPU, NVIDIA L4):

```bash
./test_sandbox.sh
```

Requires Docker.

## Stack

Python · PyTorch · Ultralytics YOLOv11 · ONNX Runtime · torchvision

## License

[MIT](./LICENSE)
