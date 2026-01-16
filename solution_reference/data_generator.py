"""
Reference Solution: Data Generator for Student Performance Predictor
Instructor Use Only
"""

import numpy as np
import pandas as pd


class StudentDataGenerator:
    """Generates synthetic student performance data."""

    def __init__(self, seed=42):
        """
        Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)

    def generate_data(self, n_samples=100, noise_std=8, outlier_pct=5):
        """
        Generate synthetic student performance data.

        The relationship is: Score = 40 + 6 * study_hours + noise
        This means:
        - Base score (no study): 40 points
        - Each hour of study adds 6 points on average
        - Noise represents individual variation

        Args:
            n_samples: Number of students
            noise_std: Standard deviation of noise (typical: 5-10)
            outlier_pct: Percentage of outliers (0-100)

        Returns:
            study_hours: Array of study hours (2-10 hours)
            exam_scores: Array of exam scores (0-100)
        """
        # Generate study hours (2-10 hours per day)
        study_hours = np.random.uniform(2, 10, n_samples)

        # Generate exam scores with linear relationship + noise
        base_score = 40
        score_per_hour = 6
        noise = np.random.normal(0, noise_std, n_samples)

        exam_scores = base_score + score_per_hour * study_hours + noise

        # Add outliers
        if outlier_pct > 0:
            n_outliers = int(n_samples * outlier_pct / 100)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)

            for idx in outlier_indices:
                outlier_type = np.random.choice(['high', 'low', 'extreme'])

                if outlier_type == 'high':
                    # Student performed much better than expected
                    exam_scores[idx] += np.random.uniform(15, 25)
                elif outlier_type == 'low':
                    # Student performed much worse than expected
                    exam_scores[idx] -= np.random.uniform(15, 25)
                else:  # extreme
                    # Very extreme case
                    exam_scores[idx] += np.random.choice([-1, 1]) * np.random.uniform(25, 35)

        # Clip scores to valid range [0, 100]
        exam_scores = np.clip(exam_scores, 0, 100)

        return study_hours, exam_scores

    def create_dataframe(self, study_hours, exam_scores):
        """
        Create a pandas DataFrame from the data.

        Args:
            study_hours: Array of study hours
            exam_scores: Array of exam scores

        Returns:
            DataFrame with columns: Study Hours, Exam Score
        """
        return pd.DataFrame({
            'Study Hours': study_hours,
            'Exam Score': exam_scores
        })

    def get_statistics(self, study_hours, exam_scores):
        """
        Calculate summary statistics.

        Args:
            study_hours: Array of study hours
            exam_scores: Array of exam scores

        Returns:
            Dictionary with statistics
        """
        correlation = np.corrcoef(study_hours, exam_scores)[0, 1]

        return {
            'n_students': len(study_hours),
            'mean_study_hours': study_hours.mean(),
            'std_study_hours': study_hours.std(),
            'mean_exam_score': exam_scores.mean(),
            'std_exam_score': exam_scores.std(),
            'min_score': exam_scores.min(),
            'max_score': exam_scores.max(),
            'correlation': correlation
        }
