"""
Batch inference for RT-DETRv4 with optional anomaly-guided fusion.
"""

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.ops as ops

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from engine.core import YAMLConfig
from engine.rtv4.anomaly_proposals import generate_anomaly_proposals, fuse_detections_with_proposals


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def apply_anomaly_mode(cfg, anomaly_mode):
    if anomaly_mode is None:
        return

    enable_anomaly = anomaly_mode == "on"

    if "HybridEncoder" in cfg.yaml_cfg:
        cfg.yaml_cfg["HybridEncoder"]["anomaly_guided"] = enable_anomaly


def load_checkpoint_state(resume_path):
    checkpoint = torch.load(resume_path, map_location="cpu")
    if "ema" in checkpoint:
        return checkpoint["ema"]["module"]
    return checkpoint["model"]


def infer_num_classes_from_state(state_dict):
    bias = state_dict.get("decoder.dec_score_head.0.bias")
    if bias is not None and bias.ndim == 1:
        return int(bias.numel())

    weight = state_dict.get("decoder.denoising_class_embed.weight")
    if weight is not None and weight.ndim == 2:
        return int(weight.shape[0] - 1)

    return None


def validate_checkpoint_compatibility(model_state, checkpoint_state, config_path, resume_path):
    model_num_classes = infer_num_classes_from_state(model_state)
    checkpoint_num_classes = infer_num_classes_from_state(checkpoint_state)

    if model_num_classes is None or checkpoint_num_classes is None:
        return

    if model_num_classes == checkpoint_num_classes:
        return

    raise RuntimeError(
        "Checkpoint/class-head mismatch detected.\n"
        f"  config: {config_path}\n"
        f"  checkpoint: {resume_path}\n"
        f"  model num_classes: {model_num_classes}\n"
        f"  checkpoint num_classes: {checkpoint_num_classes}\n"
        "Please use a matching config/checkpoint pair."
    )


def load_model(cfg, resume_path, device):
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    state = load_checkpoint_state(resume_path)
    config_path = getattr(cfg, "yaml_path", "<unknown_config>")
    validate_checkpoint_compatibility(cfg.model.state_dict(), state, config_path, resume_path)
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes, anomaly_maps=None):
            outputs = self.model(images, anomaly_maps=anomaly_maps)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model().to(device).eval()


def get_image_paths(input_dir):
    image_paths = []
    for path in sorted(Path(input_dir).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths.append(path)
    return image_paths


def anomaly_path_from_image(image_path, anomaly_dir, anomaly_suffix):
    if anomaly_dir is None:
        return None
    return Path(anomaly_dir) / f"{image_path.stem}{anomaly_suffix}"


def load_anomaly_map(anomaly_map_path, image_size, device):
    anomaly_map = np.load(anomaly_map_path, allow_pickle=False)
    anomaly_map = torch.as_tensor(anomaly_map, dtype=torch.float32)
    if anomaly_map.ndim == 2:
        anomaly_map = anomaly_map.unsqueeze(0).unsqueeze(0)
    elif anomaly_map.ndim == 3:
        if anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map.unsqueeze(0)
        else:
            anomaly_map = anomaly_map.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported anomaly map shape: {tuple(anomaly_map.shape)}")

    a_min = anomaly_map.amin(dim=(-2, -1), keepdim=True)
    a_max = anomaly_map.amax(dim=(-2, -1), keepdim=True)
    anomaly_map = (anomaly_map - a_min) / (a_max - a_min + 1e-6)
    anomaly_map = F.interpolate(anomaly_map, size=image_size, mode="bilinear", align_corners=False)
    return anomaly_map.to(device)


def preprocess_image(image_path, imgsz, device):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    transform = T.Compose([T.Resize((imgsz, imgsz)), T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=device)
    return image, image_tensor, orig_size


def xyxy_to_yolo_xywh(boxes, width, height):
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    cx = ((x1 + x2) * 0.5) / width
    cy = ((y1 + y2) * 0.5) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return torch.stack([cx, cy, bw, bh], dim=-1)


def clamp_yolo_boxes(boxes):
    return boxes.clamp_(0.0, 1.0)


def apply_nms(labels, boxes, scores, iou_thres, class_agnostic):
    if boxes.numel() == 0:
        return labels, boxes, scores

    if class_agnostic:
        keep = ops.nms(boxes, scores, iou_thres)
    else:
        keep = ops.batched_nms(boxes, scores, labels, iou_thres)

    return labels[keep], boxes[keep], scores[keep]


def filter_by_source(labels, boxes, scores, sources, keep_source):
    if sources.numel() == 0:
        return labels, boxes, scores, sources
    keep = sources == int(keep_source)
    return labels[keep], boxes[keep], scores[keep], sources[keep]


def save_txt_predictions(txt_path, labels, yolo_boxes, scores, save_conf):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for label, box, score in zip(labels.tolist(), yolo_boxes.tolist(), scores.tolist()):
            line = f"{label} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
            if save_conf:
                line += f" {score:.6f}"
            f.write(line + "\n")


def load_class_names(path):
    if path is None:
        return None

    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names


def load_class_id_map(map_path):
    if map_path is None:
        return None

    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def remap_labels(labels, class_id_map):
    if class_id_map is None or labels.numel() == 0:
        return labels

    mapped = []
    missing = set()
    for label in labels.tolist():
        if label not in class_id_map:
            missing.add(label)
            mapped.append(label)
        else:
            mapped.append(class_id_map[label])

    if missing:
        raise KeyError(f"Missing class-id mapping for labels: {sorted(missing)}")

    return torch.as_tensor(mapped, dtype=labels.dtype)


def compute_anomaly_stats(anomaly_array, box_xyxy, image_size):
    if anomaly_array is None:
        return None

    anomaly_array = np.asarray(anomaly_array, dtype=np.float32)
    if anomaly_array.ndim == 3:
        anomaly_array = anomaly_array.squeeze()
    map_h, map_w = anomaly_array.shape
    image_w, image_h = image_size

    x1, y1, x2, y2 = box_xyxy
    sx1 = int(np.clip(np.floor(x1 / image_w * map_w), 0, max(0, map_w - 1)))
    sy1 = int(np.clip(np.floor(y1 / image_h * map_h), 0, max(0, map_h - 1)))
    sx2 = int(np.clip(np.ceil(x2 / image_w * map_w), sx1 + 1, map_w))
    sy2 = int(np.clip(np.ceil(y2 / image_h * map_h), sy1 + 1, map_h))
    region = anomaly_array[sy1:sy2, sx1:sx2]
    if region.size == 0:
        return {"mean": 0.0, "max": 0.0, "min": 0.0, "area_px": 0}
    return {
        "mean": float(region.mean()),
        "max": float(region.max()),
        "min": float(region.min()),
        "area_px": int(region.size),
    }


def export_proposals_json(
    proposal_json_path,
    image_path,
    image_size,
    anomaly_path,
    labels,
    boxes,
    scores,
    yolo_boxes,
    sources,
    anomaly_array,
):
    proposal_json_path.parent.mkdir(parents=True, exist_ok=True)
    source_names = {0: "det", 1: "prop"}
    proposals = []
    for idx, (label, box_xyxy, score, box_xywh_norm, source) in enumerate(
        zip(labels.tolist(), boxes.tolist(), scores.tolist(), yolo_boxes.tolist(), sources.tolist())
    ):
        proposals.append(
            {
                "proposal_id": idx,
                "label": int(label),
                "score": float(score),
                "source": source_names.get(int(source), f"src_{int(source)}"),
                "box_xyxy": [float(v) for v in box_xyxy],
                "box_xywh_norm": [float(v) for v in box_xywh_norm],
                "anomaly_stats": compute_anomaly_stats(anomaly_array, box_xyxy, image_size),
            }
        )

    doc = {
        "image": {
            "path": str(image_path),
            "stem": Path(image_path).stem,
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "anomaly_map_path": str(anomaly_path) if anomaly_path is not None else None,
        "num_proposals": len(proposals),
        "proposals": proposals,
    }
    with open(proposal_json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)


def draw_predictions(image, labels, boxes, scores, sources=None, class_names=None):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    if sources is None:
        sources = torch.zeros((scores.numel(),), dtype=torch.int64)
    for label, box, score, source in zip(labels.tolist(), boxes.tolist(), scores.tolist(), sources.tolist()):
        color = "red" if source == 0 else "cyan"
        prefix = "det" if source == 0 else "prop"
        draw.rectangle(box, outline=color, width=2)
        if class_names is not None and 0 <= label < len(class_names):
            label_text = class_names[label]
        else:
            label_text = str(label)
        draw.text((box[0], box[1]), f"{prefix}:{label_text} {score:.2f}", fill="yellow")
    return canvas


def main(args):
    cfg = YAMLConfig(args.config, device=args.device)
    apply_anomaly_mode(cfg, args.anomaly_mode)
    model = load_model(cfg, args.resume, args.device)
    class_names = load_class_names(args.class_names)
    class_id_map = load_class_id_map(args.class_id_map)

    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")

    output_dir = Path(args.output_dir)
    label_dir = output_dir / "labels"
    vis_dir = output_dir / "vis"
    proposal_dir = output_dir / "proposals"
    proposal_manifest = output_dir / "proposal_manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    if args.export_proposals:
        proposal_dir.mkdir(parents=True, exist_ok=True)

    manifest_fp = open(proposal_manifest, "w", encoding="utf-8") if args.export_proposals else None

    processed = 0
    missing_anomaly = []

    with torch.no_grad():
        for image_path in image_paths:
            image, image_tensor, orig_size = preprocess_image(image_path, args.imgsz, args.device)
            anomaly_tensor = None
            anomaly_path = None
            anomaly_array = None

            if args.anomaly_mode == "on":
                anomaly_path = anomaly_path_from_image(image_path, args.anomaly_dir, args.anomaly_suffix)
                if anomaly_path is None or not anomaly_path.exists():
                    if args.require_anomaly:
                        raise FileNotFoundError(f"Missing anomaly map for {image_path.name}: {anomaly_path}")
                    missing_anomaly.append(image_path.name)
                else:
                    anomaly_array = np.load(anomaly_path, allow_pickle=False)
                    anomaly_tensor = load_anomaly_map(anomaly_path, (args.imgsz, args.imgsz), args.device)

            labels, boxes, scores = model(image_tensor, orig_size, anomaly_maps=anomaly_tensor)
            labels = labels[0].detach().cpu()
            boxes = boxes[0].detach().cpu()
            scores = scores[0].detach().cpu()

            keep = scores >= args.conf_thres
            labels = labels[keep]
            boxes = boxes[keep]
            scores = scores[keep]

            if args.pre_nms_topk > 0 and scores.numel() > args.pre_nms_topk:
                pre_topk_scores, pre_topk_idx = torch.topk(scores, args.pre_nms_topk)
                labels = labels[pre_topk_idx]
                boxes = boxes[pre_topk_idx]
                scores = pre_topk_scores

            labels, boxes, scores = apply_nms(
                labels,
                boxes,
                scores,
                iou_thres=args.nms_iou_thres,
                class_agnostic=args.class_agnostic_nms,
            )

            if args.max_det > 0 and scores.numel() > args.max_det:
                topk_scores, topk_idx = torch.topk(scores, args.max_det)
                labels = labels[topk_idx]
                boxes = boxes[topk_idx]
                scores = topk_scores
            sources = torch.zeros((scores.numel(),), dtype=torch.int64)

            if args.use_anomaly_proposals:
                if anomaly_path is None or not Path(anomaly_path).exists():
                    raise FileNotFoundError(
                        f"Anomaly proposal fusion requires anomaly map for {image_path.name}, got {anomaly_path}"
                    )
                proposal_boxes, proposal_scores = generate_anomaly_proposals(
                    np.load(anomaly_path, allow_pickle=False),
                    image_size=(image.width, image.height),
                    thresholds=args.proposal_thresholds,
                    min_area=args.proposal_min_area,
                    topk=args.proposal_topk,
                    morph_kernel=args.proposal_morph_kernel,
                )
                labels, boxes, scores, sources = fuse_detections_with_proposals(
                    labels,
                    boxes,
                    scores,
                    proposal_boxes,
                    proposal_scores,
                    proposal_label=args.proposal_label,
                    proposal_score_scale=args.proposal_score_scale,
                    proposal_append_iou=args.proposal_append_iou,
                    nms_iou_thres=args.proposal_fusion_nms_iou,
                    class_agnostic_nms=args.class_agnostic_nms,
                    max_det=args.max_det,
                )

            if args.detector_only:
                labels, boxes, scores, sources = filter_by_source(
                    labels,
                    boxes,
                    scores,
                    sources,
                    keep_source=0,
                )
                if args.detector_only_nms_iou_thres > 0:
                    labels, boxes, scores = apply_nms(
                        labels,
                        boxes,
                        scores,
                        iou_thres=args.detector_only_nms_iou_thres,
                        class_agnostic=True,
                    )
                    sources = torch.zeros((scores.numel(),), dtype=torch.int64)
                if args.max_det > 0 and scores.numel() > args.max_det:
                    topk_scores, topk_idx = torch.topk(scores, args.max_det)
                    labels = labels[topk_idx]
                    boxes = boxes[topk_idx]
                    scores = topk_scores
                    sources = sources[topk_idx]

            labels = remap_labels(labels, class_id_map)
            yolo_boxes = xyxy_to_yolo_xywh(boxes, image.width, image.height)
            yolo_boxes = clamp_yolo_boxes(yolo_boxes)

            txt_path = label_dir / f"{image_path.stem}.txt"
            save_txt_predictions(txt_path, labels, yolo_boxes, scores, args.save_conf)

            if args.export_proposals:
                proposal_json_path = proposal_dir / f"{image_path.stem}.json"
                export_proposals_json(
                    proposal_json_path=proposal_json_path,
                    image_path=image_path,
                    image_size=(image.width, image.height),
                    anomaly_path=anomaly_path,
                    labels=labels,
                    boxes=boxes,
                    scores=scores,
                    yolo_boxes=yolo_boxes,
                    sources=sources,
                    anomaly_array=anomaly_array,
                )
                manifest_fp.write(
                    json.dumps(
                        {
                            "image_stem": image_path.stem,
                            "image_path": str(image_path),
                            "proposal_json": str(proposal_json_path),
                            "num_proposals": int(scores.numel()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if args.save_vis:
                vis_image = draw_predictions(
                    image,
                    labels,
                    boxes,
                    scores,
                    sources=sources,
                    class_names=class_names,
                )
                vis_image.save(vis_dir / image_path.name)

            processed += 1
            if processed % args.print_freq == 0 or processed == len(image_paths):
                print(f"[{processed}/{len(image_paths)}] processed {image_path.name}")

    if manifest_fp is not None:
        manifest_fp.close()

    print(f"Inference complete. Results saved to: {output_dir}")
    print(f"Label files saved to: {label_dir}")
    if args.export_proposals:
        print(f"Proposal json files saved to: {proposal_dir}")
        print(f"Proposal manifest saved to: {proposal_manifest}")
    if args.save_vis:
        print(f"Visualizations saved to: {vis_dir}")
    if missing_anomaly:
        print(f"Images without anomaly map: {len(missing_anomaly)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--anomaly-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.1)
    parser.add_argument("--nms-iou-thres", type=float, default=0.5)
    parser.add_argument("--pre-nms-topk", type=int, default=300)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--class-agnostic-nms", action="store_true")
    parser.add_argument("--use-anomaly-proposals", action="store_true")
    parser.add_argument("--proposal-thresholds", nargs="+", type=float, default=[0.55, 0.7, 0.82])
    parser.add_argument("--proposal-min-area", type=int, default=24)
    parser.add_argument("--proposal-topk", type=int, default=30)
    parser.add_argument("--proposal-morph-kernel", type=int, default=3)
    parser.add_argument("--proposal-label", type=int, default=0)
    parser.add_argument("--proposal-score-scale", type=float, default=0.75)
    parser.add_argument("--proposal-append-iou", type=float, default=0.25)
    parser.add_argument("--proposal-fusion-nms-iou", type=float, default=0.4)
    parser.add_argument("--detector-only", action="store_true")
    parser.add_argument("--detector-only-nms-iou-thres", type=float, default=0.0)
    parser.add_argument("--export-proposals", action="store_true")
    parser.set_defaults(save_conf=True)
    parser.add_argument("--no-save-conf", dest="save_conf", action="store_false")
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--class-names", type=str, default=None)
    parser.add_argument("--class-id-map", type=str, default=None)
    parser.add_argument("--require-anomaly", action="store_true")
    parser.add_argument("--anomaly-mode", type=str, choices=["on", "off"], default="on")
    parser.add_argument("--anomaly-suffix", type=str, default="_anomaly_score.npy")
    parser.add_argument("--print-freq", type=int, default=50)
    args = parser.parse_args()
    main(args)
