import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.ops import nms
import onnxruntime as ort


def letterbox(img, new_shape=1280):
    """Resize image with padding to target size, preserving aspect ratio."""
    w, h = img.size
    r = min(new_shape / w, new_shape / h)
    new_w, new_h = int(w * r), int(h * r)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (new_shape, new_shape), (114, 114, 114))
    pad_x = (new_shape - new_w) // 2
    pad_y = (new_shape - new_h) // 2
    canvas.paste(img_resized, (pad_x, pad_y))
    return canvas, r, pad_x, pad_y


def postprocess(output, conf_thres=0.25, iou_thres=0.45):
    """Process raw ONNX output [1, 5, N] into boxes, scores."""
    # output shape: [1, 5, N] -> [N, 5] = [cx, cy, w, h, conf]
    raw = output[0]
    assert raw.ndim == 3 and raw.shape[1] == 5, (
        f"Expected ONNX output shape [1, 5, N], got {raw.shape}"
    )
    pred = raw.squeeze(0).T

    # Filter by confidence
    scores = pred[:, 4]
    mask = scores > conf_thres
    pred = pred[mask]
    scores = scores[mask]

    if len(pred) == 0:
        return np.empty((0, 4)), np.empty((0,))

    # Convert cx,cy,w,h to x1,y1,x2,y2 for NMS
    boxes = np.zeros_like(pred[:, :4])
    boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2  # x1
    boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2  # y1
    boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2  # x2
    boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2  # y2

    # NMS using torchvision
    boxes_t = torch.from_numpy(boxes).float()
    scores_t = torch.from_numpy(scores).float()
    keep = nms(boxes_t, scores_t, iou_thres)
    keep = keep.numpy()

    return boxes[keep], scores[keep]


def run():
    output_file = "/output/predictions.json"

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, default="/data/images")
        parser.add_argument("--output", type=str, default="/output/predictions.json")
        args, _ = parser.parse_known_args()

        input_dir = Path(args.input)
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            input_dir = Path("data/images")

        # Load ONNX detection model
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession("best.onnx", providers=providers)
        input_name = session.get_inputs()[0].name

        # Load feature bank for classification
        feature_bank = {}
        fb_path = Path("feature_bank.json")
        if fb_path.exists():
            with open(fb_path, "r") as f:
                feature_bank = json.load(f)

        keys = list(feature_bank.keys())
        if len(keys) > 0:
            embeddings = np.array([feature_bank[k]["embedding"] for k in keys])
            categories = np.array([feature_bank[k]["category_id"] for k in keys])
        else:
            embeddings = np.zeros((1, 2048))
            categories = np.zeros((1,))

        # Load ResNet50 for classification
        feature_model = models.resnet50()
        resnet_path = Path("resnet50_offline.pth")
        if resnet_path.exists():
            feature_model.load_state_dict(
                torch.load(resnet_path, map_location="cpu", weights_only=True)
            )

        feature_model.fc = torch.nn.Identity()
        feature_model.train(False)

        classify_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        predictions = []

        for img_path in sorted(input_dir.glob("*.jpg")):
            img_id_str = img_path.stem.split("_")[-1]
            image_id = int(img_id_str) if img_id_str.isdigit() else 0

            pil_img = Image.open(str(img_path)).convert("RGB")
            orig_w, orig_h = pil_img.size

            # Preprocess for ONNX: letterbox + normalize
            lb_img, ratio, pad_x, pad_y = letterbox(pil_img, 1280)
            arr = np.array(lb_img).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # [1,3,1280,1280]

            # Run detection
            outputs = session.run(None, {input_name: arr})
            boxes, scores = postprocess(outputs, conf_thres=0.25, iou_thres=0.45)

            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]

                # Undo letterbox: remove padding, then unscale
                x1 = (x1 - pad_x) / ratio
                y1 = (y1 - pad_y) / ratio
                x2 = (x2 - pad_x) / ratio
                y2 = (y2 - pad_y) / ratio

                # Clamp to image bounds
                x1 = max(0, min(x1, orig_w))
                y1 = max(0, min(y1, orig_h))
                x2 = max(0, min(x2, orig_w))
                y2 = max(0, min(y2, orig_h))

                w = x2 - x1
                h = y2 - y1
                conf = float(scores[i])

                category_id = 0

                # Classification via ResNet50 + feature bank
                try:
                    crop = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
                    if crop.size[0] > 10 and crop.size[1] > 10:
                        if crop.mode != "RGB":
                            crop = crop.convert("RGB")
                        input_tensor = classify_transform(crop).unsqueeze(0)

                        with torch.no_grad():
                            feat = feature_model(input_tensor).squeeze().numpy()
                            feat_norm = np.linalg.norm(feat)
                            if feat_norm > 0:
                                feat = feat / feat_norm

                        if len(keys) > 0:
                            sims = np.dot(embeddings, feat)
                            best_idx = np.argmax(sims)
                            best_sim = sims[best_idx]
                            if best_sim > 0.4:
                                category_id = int(categories[best_idx])
                except Exception:
                    pass

                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [
                            round(float(x1), 1),
                            round(float(y1), 1),
                            round(float(w), 1),
                            round(float(h), 1),
                        ],
                        "score": round(conf, 3),
                    }
                )

        with open(output_file, "w") as f:
            json.dump(predictions, f)

    except Exception as e:
        print(f"Error in run(): {e}")
        with open(str(output_file), "w") as f:
            json.dump([], f)


if __name__ == "__main__":
    run()
