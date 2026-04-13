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

