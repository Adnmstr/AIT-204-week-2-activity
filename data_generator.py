import numpy as np
import pandas as pd


class StudentDataGenerator:
    """Generates synthetic student performance data."""

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)

    @staticmethod
    def heteroscedastic_scale(study_hours: np.ndarray) -> np.ndarray:
        """
        Scale factor for noise as a function of study hours.
        >1 means noisier; <1 means less noisy.
        """
        # 2..10 hours -> scale ~ 0.7 .. 1.5 (smooth, monotonic)
        x = np.asarray(study_hours).astype(float)
        return 0.6 + 0.1 * x  # at x=2 => 0.8, x=10 => 1.6

    def generate_data(self, n_samples=100, noise_std=8, outlier_pct=5, heteroscedastic=False):
        """
        Generate synthetic student performance data.

        Relationship: Score = 40 + 6 * study_hours + noise

        If heteroscedastic=True, noise std varies with study hours and we also
        return sample weights ~ 1/variance for weighted training.
        """
        study_hours = np.random.uniform(2, 10, n_samples)

        base_score = 40
        score_per_hour = 6

        if heteroscedastic:
            scale = self.heteroscedastic_scale(study_hours)
            noise_std_i = noise_std * scale
            noise = np.random.normal(0, noise_std_i, n_samples)

            # weights proportional to 1/variance
            sample_weights = 1.0 / (noise_std_i ** 2)
        else:
            noise = np.random.normal(0, noise_std, n_samples)
            sample_weights = None

        exam_scores = base_score + score_per_hour * study_hours + noise

        # Add outliers (same as before)
        if outlier_pct > 0:
            n_outliers = int(n_samples * outlier_pct / 100)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

            for idx in outlier_indices:
                outlier_type = np.random.choice(["high", "low", "extreme"])
                if outlier_type == "high":
                    exam_scores[idx] += np.random.uniform(15, 25)
                elif outlier_type == "low":
                    exam_scores[idx] -= np.random.uniform(15, 25)
                else:
                    exam_scores[idx] += np.random.choice([-1, 1]) * np.random.uniform(25, 35)

        exam_scores = np.clip(exam_scores, 0, 100)
        return study_hours, exam_scores, sample_weights

    def create_dataframe(self, study_hours, exam_scores, heteroscedastic=False, base_noise_std=8):
        df = pd.DataFrame({
            "Study Hours": study_hours,
            "Exam Score": exam_scores
        })

        if heteroscedastic:
            scale = self.heteroscedastic_scale(study_hours)
            df["Noise Scale"] = scale
            df["Noise Std (per student)"] = base_noise_std * scale

        return df

    def get_statistics(self, study_hours, exam_scores):
        correlation = np.corrcoef(study_hours, exam_scores)[0, 1]
        return {
            "n_students": len(study_hours),
            "mean_study_hours": study_hours.mean(),
            "std_study_hours": study_hours.std(),
            "mean_exam_score": exam_scores.mean(),
            "std_exam_score": exam_scores.std(),
            "min_score": exam_scores.min(),
            "max_score": exam_scores.max(),
            "correlation": correlation
        }
