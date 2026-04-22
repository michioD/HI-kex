import numpy as np
import torchvision.ops
def extract_confidence_detection(results):
    """
    Image-level confidence based on weakest detected object.
    """
    if not results or len(results[0].boxes) == 0:
        return 0.0

    confs = results[0].boxes.conf.cpu().numpy()

    # Bottleneck principle: weakest object dominates
    p_t = np.min(confs)

    return float(np.clip(p_t, 0.0, 1.0))

def weighted_confidence_detection(results):
    """
    Set all weights to 1 first for simpliciy. Extract the confidence scores of all detected objects and compute a weighted average.
    """
    if not results or len(results[0].boxes) == 0:
        return 0.0

    confs = results[0].boxes.conf.cpu().numpy()
    weights = np.ones_like(confs)

    p_t = np.sum(weights * confs) / np.sum(weights)
    return float(np.clip(p_t, 0.0, 1.0))

def extract_comprehensive_detection_metric(results, lambda_nms=0.3, lambda_fn=1.0):
    """
    Comprehensive Object Detection Metric.
    Punishes Weak Confidence, NMS Failures (FP), and Ghost Suppression (FN).
    """
    if not results or len(results[0].boxes) == 0:
        return 0.0 
        
    confs = results[0].boxes.conf
    
    # 1. SPLIT THE DATA (The Threshold Hack)
    # Assume results were generated with model.predict(conf=0.15)
    det_mask = confs >= 0.50
    ghost_mask = confs < 0.50
    
    det_boxes = results[0].boxes[det_mask]
    ghost_confs = confs[ghost_mask]
    
    # If no valid objects were found above 0.50, the scene is a failure
    if len(det_boxes) == 0:
        return 0.0

    # 2. THE WEAKEST LINK (C_min)
    c_min = det_boxes.conf.min().item()
    
    # 3. FALSE POSITIVE PENALTY (NMS Failure / Spatial Overlap)
    iou_max = 0.0
    if len(det_boxes) > 1:
        iou_matrix = torchvision.ops.box_iou(det_boxes.xyxy, det_boxes.xyxy)
        iou_matrix.fill_diagonal_(0.0)
        iou_max = iou_matrix.max().item()
        
    spatial_penalty = max(0.0, 1.0 - (lambda_nms * iou_max))
    
    # 4. FALSE NEGATIVE PENALTY (Strongest Ghost)
    g_max = 0.0
    if len(ghost_confs) > 0:
        g_max = ghost_confs.max().item()
        
    fn_penalty = max(0.0, 1.0 - (lambda_fn * g_max))
    
    # 5. COMPOSITE CONFIDENCE
    p_t = c_min * spatial_penalty * fn_penalty
    
    return float(np.clip(p_t, 0.0, 1.0))