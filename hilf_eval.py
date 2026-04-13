import numpy as np
import torch
import torchvision
import bisect
import glob
from ultralytics import YOLO
from metrics import extract_confidence_detection, weighted_confidence_detection, extract_comprehensive_detection_metric

# ============================================================
# HIL-F: Hierarchical Inference Learning (Full Feedback)
# ============================================================

class HIL_F:
    def __init__(self, n_samples, beta=0.5):
        self.boundaries = [0.0, 1.0]
        self.weights = [1.0]
        self.beta = beta

        self.eta = np.sqrt(8 * np.log(n_samples + 1) / n_samples)

    def get_decision(self, p_t):
        area_below = 0.0
        total_area = 0.0

        for i in range(len(self.weights)):
            w = self.weights[i]
            b_low = self.boundaries[i]
            b_high = self.boundaries[i + 1]

            area = w * (b_high - b_low)
            total_area += area

            if b_high <= p_t:
                area_below += area
            elif b_low < p_t < b_high:
                area_below += w * (p_t - b_low)

        q_t = area_below / total_area if total_area > 0 else 0.5
        q_t = np.clip(q_t, 0.0, 1.0)

        accept_sml = np.random.rand() < q_t
        return accept_sml, q_t

    def update(self, p_t, y_t):
        # Split interval
        if p_t not in self.boundaries:
            idx = bisect.bisect_left(self.boundaries, p_t)
            self.boundaries.insert(idx, p_t)
            self.weights.insert(idx, self.weights[idx - 1])

        # Exponential weight update
        for i in range(len(self.weights)):
            b_low = self.boundaries[i]

            if b_low >= p_t:
                loss = self.beta  # would offload
            else:
                loss = y_t        # would accept

            self.weights[i] *= np.exp(-self.eta * loss)

# ============================================================
# Confidence Metric (Detection)
# ============================================================

confidence_metric = weighted_confidence_detection

# ============================================================
# Box IoU
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


# ============================================================
# Detection Cost Function
# ============================================================
def calculate_detection_cost(sml_results, lml_results, iou_threshold=0.5):
    """
    Returns:
        0.0 -> S-ML matches L-ML
        1.0 -> mismatch
    """

    sml_boxes = sml_results[0].boxes
    lml_boxes = lml_results[0].boxes

    # Presence mismatch
    if len(sml_boxes) == 0 and len(lml_boxes) == 0:
        return 0.0
    if len(sml_boxes) == 0 or len(lml_boxes) == 0:
        return 1.0

    sml_xyxy = sml_boxes.xyxy.cpu().numpy()
    lml_xyxy = lml_boxes.xyxy.cpu().numpy()
    matched = 0
    # Match each L-ML box with best S-ML box
    for l_box in lml_xyxy:
        best_iou = 0.0

        for s_box in sml_xyxy:
            iou = box_iou(l_box, s_box)
            if iou > best_iou:
                best_iou = iou

        if best_iou >= iou_threshold:
            matched += 1

    # Require all L-ML detections to be matched
    return 0.0 if matched == len(lml_xyxy) else 1.0


# ============================================================
# Main Simulation Loop
# ============================================================

def run_hierarchical_inference_simulation(image_paths):
    print("Loading YOLOv8 detection models...")

    # Detection models (NOT segmentation)
    s_ml = YOLO('yolov8n.pt')  # Edge model
    l_ml = YOLO('yolov8x.pt')  # Server model

    n_samples = len(image_paths)
    beta = 0.5

    hil_f = HIL_F(n_samples=n_samples, beta=beta)

    total_cost = 0.0
    offloads = 0
    accepted_incorrect_cost = 0
    for t, img_path in enumerate(image_paths):
        # --- 1. Local inference ---
        sml_results = s_ml.predict(img_path, verbose=False)
        # --- 2. Confidence ---
        # p_t = extract_confidence_detection(sml_results)
        p_t = confidence_metric(sml_results)  # alternative metric
        # --- 3. Decision ---
        accept_sml, q_t = hil_f.get_decision(p_t)
        # --- 4. Execute + evaluate ---
        lml_results = l_ml.predict(img_path, verbose=False)
        y_t = calculate_detection_cost(sml_results, lml_results)
            
        if accept_sml:
            step_cost = y_t
            action = "ACCEPTED "
            accepted_incorrect_cost += y_t
        else:
            step_cost = beta
            offloads += 1
            action = "OFFLOADED"
        total_cost += step_cost
        # --- 5. Update ---
        hil_f.update(p_t, y_t)

        print(f"Sample {t+1:02d}/{n_samples} | p_t: {p_t:.3f} | q_t: {q_t:.3f} | Action: {action} | Y_t: {y_t}")

    print("\n--- Simulation Complete ---")
    print(f"Total Images: {n_samples}")
    print(f"Offloaded: {offloads} ({(offloads/n_samples)*100:.1f}%)")
    print(f"Average Cost: {total_cost / n_samples:.3f}")
    # """Calculate the accuracy score i.e out of all the accpted samples how many were correct (y_t = 0)"""

    accepted_samples = n_samples - offloads
    correct_accepts = accepted_samples - accepted_incorrect_cost
    accuracy = correct_accepts / accepted_samples if accepted_samples > 0 else 0.0
    print(f"Accuracy of accepted samples: {accuracy*100:.1f}%")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[0:50]

    if not image_paths:
        print("Error: No images found.")
        quit()
    else:
        print(f"Loaded {len(image_paths)} images. Starting simulation...\n")
    
        print("Loading YOLOv8 detection models...")

    # Detection models (NOT segmentation)
    s_ml = YOLO('yolov8n.pt')  # Edge model
    l_ml = YOLO('yolov8x.pt')  # Server model

    n_samples = len(image_paths)
    beta = 0.5

    hil_f = HIL_F(n_samples=n_samples, beta=beta)

    total_cost = 0.0
    offloads = 0
    accepted_incorrect_cost = 0
    incorrect_count = 0
    for t, img_path in enumerate(image_paths):
        # --- 1. Local inference ---
        sml_results = s_ml.predict(img_path, verbose=False)
        # --- 2. Confidence ---
        # p_t = extract_confidence_detection(sml_results)
        p_t = confidence_metric(sml_results)  # alternative metric
        # --- 3. Decision ---
        accept_sml, q_t = hil_f.get_decision(p_t)
        # --- 4. Execute + evaluate ---
        lml_results = l_ml.predict(img_path, verbose=False)
        y_t = calculate_detection_cost(sml_results, lml_results)
        incorrect_count += y_t
        if accept_sml:
            step_cost = y_t
            action = "ACCEPTED "
            accepted_incorrect_cost += y_t
        else:
            step_cost = beta
            offloads += 1
            action = "OFFLOADED"
        total_cost += step_cost
        # --- 5. Update ---
        hil_f.update(p_t, y_t)

        print(f"Sample {t+1:02d}/{n_samples} | p_t: {p_t:.3f} | q_t: {q_t:.3f} | Action: {action} | Y_t: {y_t}")

    print("\n--- Simulation Complete ---")
    print(f"Total Images: {n_samples}")
    print(f"Offloaded: {offloads} ({(offloads/n_samples)*100:.1f}%)")
    print(f"Average Cost: {total_cost / n_samples:.3f}")
    # print(f"Incorrect Predictions: {incorrect_count}")
    # """Calculate the accuracy score i.e out of all the accpted samples how many were correct (y_t = 0)"""

    accepted_samples = n_samples - offloads
    correct_accepts = accepted_samples - accepted_incorrect_cost
    accuracy = correct_accepts / accepted_samples if accepted_samples > 0 else 0.0
    print(f"Accuracy of accepted samples: {accuracy*100:.1f}%")

    accuracy_of_sml = (n_samples - incorrect_count) / n_samples 
    print(f"Overall S-ML Accuracy: {accuracy_of_sml*100:.1f}%")
