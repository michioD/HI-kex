import csv
import os
import numpy as np
from metrics import *
import glob
# from ultralytics import YOLO
from precompute import check_cache, get_cached_data
from HIL_F_Vanilla import HIL_F_Vanilla
# check_cache()  # Ensure cache is ready before evaluation


# ============================================================
# Continuous Detection Cost Function (1 - F1 Score)
# ============================================================
def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

def calculate_detection_cost_f1(sml_results, lml_results, iou_threshold=0.45):
    """
    Returns (1.0 - F1_Score).
    - 0.0: Perfect match (Precision=1, Recall=1)
    - 1.0: Complete failure
    """
    sml_boxes = sml_results[0].boxes
    lml_boxes = lml_results[0].boxes

    if len(sml_boxes) == 0 and len(lml_boxes) == 0:
        return 0.0
    if len(sml_boxes) == 0 or len(lml_boxes) == 0:
        return 1.0

    sml_xyxy = sml_boxes.xyxy.cpu().numpy()
    sml_cls = sml_boxes.cls.cpu().numpy()
    lml_xyxy = lml_boxes.xyxy.cpu().numpy()
    lml_cls = lml_boxes.cls.cpu().numpy()

    tp = 0
    matched_sml = set()

    for i, l_box in enumerate(lml_xyxy):
        l_class = lml_cls[i]
        best_iou = 0
        best_j = -1
        for j, s_box in enumerate(sml_xyxy):
            if j in matched_sml: continue
            iou = box_iou(l_box, s_box)
            if iou > best_iou and sml_cls[j] == l_class:
                best_iou = iou
                best_j = j
        
        if best_iou >= iou_threshold:
            tp += 1
            matched_sml.add(best_j)

    precision = tp / len(sml_xyxy) if len(sml_xyxy) > 0 else 0
    recall = tp / len(lml_xyxy) if len(lml_xyxy) > 0 else 0
    
    if (precision + recall) == 0:
        return 1.0
        
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(1.0 - f1)


def calculate_detection_cost_full(sml_results, lml_results, iou_threshold=0.5):
    """
    Step-by-step Strict Cost Function:
    1. Cardinality check (N_s == N_l)
    2. Spatial matching for every L-ML box to a unique S-ML box
    3. Classification check for every matched pair
    4. Verify no extra/unmatched boxes in S-ML
    """
    s_boxes = sml_results[0].boxes
    l_boxes = lml_results[0].boxes
    Y_t = 0.0
    fp = 0.0
    fn = 0.0
    misclassified = 0.0
    
    # 1. Cardinality Check
    if len(s_boxes) != len(l_boxes):
        return 1.0, 0.0, 0.0, 0.0
    
    # Base case: both empty is a success
    if len(s_boxes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    s_xyxy = s_boxes.xyxy.cpu().numpy()
    s_cls = s_boxes.cls.cpu().numpy()
    l_xyxy = l_boxes.xyxy.cpu().numpy()
    l_cls = l_boxes.cls.cpu().numpy()

    matched_s_indices = set()

    # 2. Find a matching for each box in L-ML
    for i in range(len(l_xyxy)):
        best_iou = -1.0
        match_idx = -1
        
        for j in range(len(s_xyxy)):
            if j in matched_s_indices:
                continue
                
            iou = box_iou(l_xyxy[i], s_xyxy[j])
            if iou > best_iou:
                best_iou = iou
                match_idx = j

        # Check if spatial match exists
        if match_idx == -1 or best_iou < iou_threshold:
            return 1.0, 0.0, 1.0, 0.0 # Error: Missing detection
            
        # 3. Check if classification matches for the pair
        if s_cls[match_idx] != l_cls[i]:
            misclassified = 1.0
            Y_t = 1.0
            
        matched_s_indices.add(match_idx)

    # 4. Check for extra boxes in S-ML
    if len(matched_s_indices) != len(s_xyxy):
        fp = 1.0
        Y_t = 1.0

    return Y_t, fp, fn, misclassified



# confidence_metric = random_confidence_metric = lambda cached_data: np.random.rand()  # Placeholder for actual confidence metric
confidence_metric = weakest_link_confidence

def run_hierarchical_inference_simulation(image_paths, output_csv="hilf_results.csv"):
    n_samples = len(image_paths)
    beta = 0.5
    target = 0.8
    # hil_f = HIL_F(n_samples=n_samples, beta=beta, target_accuracy=target)
    hil_f = HIL_F_Vanilla(n_samples=n_samples, beta=beta)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "p_t", "q_t", "Action", "Y_t", "Success", "FP", "FN", "Misclassified"])

    total_cost = 0.0
    offloads = 0
    accepted_cost = 0
    accepted_count = 0
    correct_identifications = 0
    decision_success = 0
    correct_offload = 0
    incorrect_offload = 0
    correct_accept = 0
    incorrect_accept = 0
    for t, img_path in enumerate(image_paths):
        cached_data = get_cached_data(img_path)
        Y_t, fp, fn, misclassified = calculate_detection_cost_full(cached_data['yolov8n_coco'], cached_data['yolov8x_coco'])
        p_t = confidence_metric(cached_data['yolov8n_coco'])
        
        accept_sml, q_t = hil_f.get_decision(p_t)
        
        if accept_sml:
            step_cost = Y_t
            action = "ACCEPTED"
            accepted_cost += Y_t
            accepted_count += 1
            if Y_t == 0:
                correct_identifications += 1
        elif not accept_sml:
            step_cost = beta
            offloads += 1
            action = "OFFLOADED"
            correct_identifications += 1 # Assumed correct via Oracle L-ML

        total_cost += step_cost
        hil_f.update(p_t, Y_t)

        decision_success += 1 if (Y_t > 0 and not accept_sml) or (Y_t == 0 and accept_sml) else 0

        correct_accept += 1 if accept_sml and Y_t == 0 else 0
        incorrect_accept += 1 if accept_sml and Y_t > 0 else 0
        correct_offload += 1 if not accept_sml and Y_t > 0 else 0
        incorrect_offload += 1 if not accept_sml and Y_t == 0 else 0
        
        success = "Yes" if Y_t == 0 else "No"
        
        if (t + 1) % 100 == 0 or t == 0:
            print(f"Sample {t+1:04d} | p_t: {p_t:.3f} | q_t: {q_t:.3f} | Action: {action:<9} | Cost: {Y_t} | Success: {success}")
            
        with open(output_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([t, p_t, q_t, action, Y_t, success, fp, fn, misclassified])
    
    print("\n--- Simulation Complete ---")
    print(f"Overall System Accuracy: {(correct_identifications/n_samples)*100:.1f}%")
    if accepted_count > 0:
        avg_acc = 1.0 - (accepted_cost / accepted_count)
        print(f"Accepted Sample Accuracy (Local): {avg_acc*100:.1f}%")
    print(f"Offloading Rate: {(offloads/n_samples)*100:.1f}%")
    
    if n_samples > 0:
        decision_accuracy = (decision_success / n_samples) * 100
        print(f"Decision Accuracy (Optimal Choice): {decision_accuracy:.1f}%")
    
    # average cost per sample
    avg_cost = total_cost / n_samples if n_samples > 0 else 0.
    print(f"Average Cost per Sample: {avg_cost:.3f}")

    # number of images correctly offloaded
    print(f"Correct Offloads: {correct_offload}")
    # number of images incorrectly offloaded
    print(f"Incorrect Offloads: {incorrect_offload}")
    # number of images correctly accepted
    print(f"Correct Accepts: {correct_accept}")
    # number of images incorrectly accepted
    print(f"Incorrect Accepts: {incorrect_accept}")


if __name__ == "__main__":
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[0:5000]
    run_hierarchical_inference_simulation(image_paths)
