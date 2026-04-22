"""
Utilities for generating anomaly-map proposals and fusing them with detector outputs.
"""

from typing import Iterable, Tuple

import cv2
import numpy as np
import torch
import torchvision.ops as ops


def _to_numpy_map(anomaly_map) -> np.ndarray:
    if anomaly_map is None:
        raise ValueError("anomaly_map must not be None")

    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().float().cpu()
        if anomaly_map.ndim == 4:
            anomaly_map = anomaly_map[0, 0]
        elif anomaly_map.ndim == 3:
            anomaly_map = anomaly_map[0]
        elif anomaly_map.ndim != 2:
            raise ValueError(f"Unsupported anomaly tensor shape: {tuple(anomaly_map.shape)}")
        anomaly_map = anomaly_map.numpy()
    else:
        anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
        if anomaly_map.ndim != 2:
            raise ValueError(f"Unsupported anomaly array shape: {tuple(anomaly_map.shape)}")

    anomaly_map = anomaly_map.astype(np.float32)
    a_min = float(anomaly_map.min())
    a_max = float(anomaly_map.max())
    if a_max > a_min:
        anomaly_map = (anomaly_map - a_min) / (a_max - a_min)
    else:
        anomaly_map = np.zeros_like(anomaly_map, dtype=np.float32)
    return anomaly_map


def generate_anomaly_proposals(
    anomaly_map,
    image_size: Tuple[int, int],
    thresholds: Iterable[float] = (0.55, 0.7, 0.82),
    min_area: int = 24,
    topk: int = 30,
    morph_kernel: int = 3,
):
    """
    Returns boxes in xyxy pixel coordinates and proposal scores.
    """
    anomaly_map = _to_numpy_map(anomaly_map)
    image_w, image_h = image_size
    map_h, map_w = anomaly_map.shape
    anomaly_map = cv2.GaussianBlur(anomaly_map, (5, 5), 0)

    kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8) if morph_kernel > 1 else None
    min_short_side = max(4.0, min(image_w, image_h) * 0.01)
    min_long_side = max(10.0, min(image_w, image_h) * 0.03)
    min_box_area = max(float(min_area), min_short_side * min_long_side)

    proposal_boxes = []
    proposal_scores = []

    for threshold in thresholds:
        binary = (anomaly_map >= float(threshold)).astype(np.uint8)
        if kernel is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label_idx in range(1, num_labels):
            x, y, w, h, area = stats[label_idx]
            if area < min_area or w <= 0 or h <= 0:
                continue

            component = anomaly_map[labels == label_idx]
            mean_score = float(component.mean())
            max_score = float(component.max())
            score = 0.7 * max_score + 0.3 * mean_score

            x1 = x * image_w / map_w
            y1 = y * image_h / map_h
            x2 = (x + w) * image_w / map_w
            y2 = (y + h) * image_h / map_h
            box_w = x2 - x1
            box_h = y2 - y1
            short_side = min(box_w, box_h)
            long_side = max(box_w, box_h)
            box_area = box_w * box_h

            # Remove tiny fragmented responses while keeping long scratch-like regions.
            if short_side < min_short_side:
                continue
            if long_side < min_long_side:
                continue
            if box_area < min_box_area:
                continue

            # Fuzzy low-contrast connected components are common false positives on normal texture.
            if max_score < min(0.98, float(threshold) + 0.05):
                continue
            if mean_score < float(threshold) * 0.78:
                continue

            proposal_boxes.append([x1, y1, x2, y2])
            proposal_scores.append(score)

    if not proposal_boxes:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
        )

    boxes = torch.tensor(proposal_boxes, dtype=torch.float32)
    scores = torch.tensor(proposal_scores, dtype=torch.float32)

    keep = ops.nms(boxes, scores, 0.5)
    boxes = boxes[keep]
    scores = scores[keep]

    if topk > 0 and scores.numel() > topk:
        scores, topk_idx = torch.topk(scores, topk)
        boxes = boxes[topk_idx]

    return boxes, scores


def fuse_detections_with_proposals(
    det_labels: torch.Tensor,
    det_boxes: torch.Tensor,
    det_scores: torch.Tensor,
    proposal_boxes: torch.Tensor,
    proposal_scores: torch.Tensor,
    proposal_label: int = 0,
    proposal_score_scale: float = 0.75,
    proposal_append_iou: float = 0.25,
    nms_iou_thres: float = 0.5,
    class_agnostic_nms: bool = True,
    max_det: int = 300,
):
    if proposal_boxes.numel() == 0:
        source = torch.zeros((det_scores.numel(),), dtype=torch.int64)
        return det_labels, det_boxes, det_scores, source

    if det_boxes.numel() == 0:
        fused_labels = torch.full((proposal_boxes.shape[0],), proposal_label, dtype=torch.int64)
        fused_scores = proposal_scores * proposal_score_scale
        source = torch.ones((proposal_boxes.shape[0],), dtype=torch.int64)
        return fused_labels, proposal_boxes, fused_scores, source

    proposal_scores = proposal_scores * proposal_score_scale
    proposal_labels = torch.full((proposal_boxes.shape[0],), proposal_label, dtype=torch.int64)

    ious = ops.box_iou(proposal_boxes, det_boxes)
    keep_prop = ious.max(dim=1).values < proposal_append_iou

    proposal_boxes = proposal_boxes[keep_prop]
    proposal_scores = proposal_scores[keep_prop]
    proposal_labels = proposal_labels[keep_prop]

    if proposal_boxes.numel() == 0:
        source = torch.zeros((det_scores.numel(),), dtype=torch.int64)
        return det_labels, det_boxes, det_scores, source

    labels = torch.cat([det_labels, proposal_labels], dim=0)
    boxes = torch.cat([det_boxes, proposal_boxes], dim=0)
    scores = torch.cat([det_scores, proposal_scores], dim=0)
    source = torch.cat(
        [
            torch.zeros((det_scores.numel(),), dtype=torch.int64),
            torch.ones((proposal_scores.numel(),), dtype=torch.int64),
        ],
        dim=0,
    )

    if class_agnostic_nms:
        keep = ops.nms(boxes, scores, nms_iou_thres)
    else:
        keep = ops.batched_nms(boxes, scores, labels, nms_iou_thres)

    labels = labels[keep]
    boxes = boxes[keep]
    scores = scores[keep]
    source = source[keep]

    if max_det > 0 and scores.numel() > max_det:
        scores, topk_idx = torch.topk(scores, max_det)
        labels = labels[topk_idx]
        boxes = boxes[topk_idx]
        source = source[topk_idx]

    return labels, boxes, scores, source
