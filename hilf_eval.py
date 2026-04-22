import numpy as np
import torch
import os
import torchvision
import bisect
import glob
from ultralytics import YOLO
from precompute import check_cache, get_cached_data
check_cache()  # Ensure cache is ready before evaluation


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

        for i in range(len(self.weights)):
            b_low = self.boundaries[i]
            if b_low >= p_t:
                loss = self.beta 
            else:
                loss = y_t

            self.weights[i] *= np.exp(-self.eta * loss)


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
def calculate_detection_cost(sml_results, lml_results, iou_threshold=0.45):
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

def weakest_link_confidence(results):
    """
    Image-level confidence based on weakest detected object.
    """
    if not results or len(results[0].boxes) == 0:
        return 0.0

    confs = results[0].boxes.conf.cpu().numpy()
    p_t = np.min(confs)
    return float(p_t)

confidence_metric = weakest_link_confidence

def run_hierarchical_inference_simulation(image_paths):
    device = torch.device("mps")

    n_samples = len(image_paths)
    beta = 0.5

    hil_f = HIL_F(n_samples=n_samples, beta=beta)

    total_cost = 0.0
    offloads = 0
    accepted_incorrect_cost = 0
    incorrect_count = 0
    for t, img_path in enumerate(image_paths):
        cached_data = get_cached_data(img_path)
        sml_results = cached_data['yolov8n_coco']
        lml_results = cached_data['yolov8x_coco']
        y_t = calculate_detection_cost(sml_results, lml_results)
        p_t = confidence_metric(sml_results) 
        accept_sml, q_t = hil_f.get_decision(p_t)
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

        hil_f.update(p_t, y_t)
        if accept_sml and y_t == 0:
            choice = "Good"
        elif not accept_sml and y_t == 1:
            choice = "Good"
        else:
            choice = "Bad"

        print(f"Sample {t+1:02d}/{n_samples} | p_t: {p_t:.3f} | q_t: {q_t:.3f} | Action: {action} | Y_t: {y_t} | {choice}" )

    print(f"Total Images: {n_samples}")
    print(f"Offloaded: {offloads} ({(offloads/n_samples)*100:.1f}%)")
    print(f"Average Cost: {total_cost / n_samples:.3f}")
    accepted_samples = n_samples - offloads
    correct_accepts = accepted_samples - accepted_incorrect_cost
    accuracy = correct_accepts / accepted_samples if accepted_samples > 0 else 0.0
    print(f"Accuracy of accepted samples: {accuracy*100:.1f}%")
    
    accuracy_of_sml = (n_samples - incorrect_count) / n_samples 
    print(f"S-ML accuracy on this dataset: {accuracy_of_sml*100:.1f}%")


if __name__ == "__main__":
    image_paths = sorted(glob.glob("datasets/coco_images/val2017/*.jpg"))[0:5000]
    run_hierarchical_inference_simulation(image_paths)