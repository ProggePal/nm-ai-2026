/// Scoring: entropy-weighted KL divergence (matches Python src/scoring.py).

use crate::types::NUM_CLASSES;

/// Score a prediction against ground truth.
/// Returns the competition score (0-100, higher is better).
pub fn score_prediction(ground_truth: &[f64], prediction: &[f64], height: usize, width: usize) -> f64 {
    // Must match Python src/constants.py ENTROPY_THRESHOLD
    let entropy_threshold = 0.01;
    let mut weighted_kl_sum = 0.0;
    let mut entropy_sum = 0.0;

    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * NUM_CLASSES;

            // Compute entropy of ground truth cell
            let mut cell_entropy = 0.0;
            for c in 0..NUM_CLASSES {
                let p = ground_truth[base + c];
                if p > 1e-15 {
                    cell_entropy -= p * p.ln();
                }
            }

            if cell_entropy <= entropy_threshold {
                continue;
            }

            // Compute KL divergence for this cell
            let mut cell_kl = 0.0;
            for c in 0..NUM_CLASSES {
                let p = ground_truth[base + c].max(1e-15);
                let q = prediction[base + c].max(1e-15);
                cell_kl += p * (p / q).ln();
            }

            weighted_kl_sum += cell_entropy * cell_kl;
            entropy_sum += cell_entropy;
        }
    }

    if entropy_sum <= 0.0 {
        return 100.0;
    }

    let weighted_kl = weighted_kl_sum / entropy_sum;
    let score = 100.0 * (-3.0 * weighted_kl).exp();
    score.max(0.0).min(100.0)
}
