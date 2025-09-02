#!/usr/bin/env python3
"""
Evaluation Metrics for Medical Image Analysis
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import tensorflow as tf


class ClassificationMetrics:
    """Comprehensive classification metrics for medical images"""

    def __init__(self, y_true: List[int], y_pred: List[int], 
                 class_names: Optional[List[str]] = None,
                 y_prob: Optional[np.ndarray] = None):

        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = y_prob
        self.class_names = class_names or [f'Class_{i}' for i in range(len(np.unique(y_true)))]
        self.num_classes = len(self.class_names)

    def calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        return {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(self.y_true, self.y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
        }

    def calculate_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics"""
        precision_per_class = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)

        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i])
            }

        return per_class_metrics

    def calculate_auc_metrics(self) -> Dict[str, float]:
        """Calculate AUC metrics if probabilities are available"""
        if self.y_prob is None:
            return {'auc_macro': None, 'auc_weighted': None}

        try:
            if self.num_classes == 2:
                # Binary classification
                auc = roc_auc_score(self.y_true, self.y_prob[:, 1])
                return {'auc': float(auc)}
            else:
                # Multi-class classification
                y_true_bin = label_binarize(self.y_true, classes=range(self.num_classes))
                auc_macro = roc_auc_score(y_true_bin, self.y_prob, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(y_true_bin, self.y_prob, average='weighted', multi_class='ovr')

                return {
                    'auc_macro': float(auc_macro),
                    'auc_weighted': float(auc_weighted)
                }
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            return {'auc_macro': None, 'auc_weighted': None}

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        return confusion_matrix(self.y_true, self.y_pred)

    def get_classification_report(self) -> str:
        """Get detailed classification report"""
        return classification_report(
            self.y_true, 
            self.y_pred, 
            target_names=self.class_names,
            zero_division=0
        )

    def calculate_sensitivity_specificity(self) -> Dict[str, Dict[str, float]]:
        """Calculate sensitivity and specificity for each class"""
        cm = self.get_confusion_matrix()

        results = {}
        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            results[class_name] = {
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'true_positives': int(tp),
                'false_negatives': int(fn),
                'false_positives': int(fp),
                'true_negatives': int(tn)
            }

        return results

    def calculate_all_metrics(self) -> Dict:
        """Calculate all available metrics"""
        metrics = {
            'basic_metrics': self.calculate_basic_metrics(),
            'per_class_metrics': self.calculate_per_class_metrics(),
            'auc_metrics': self.calculate_auc_metrics(),
            'sensitivity_specificity': self.calculate_sensitivity_specificity(),
            'confusion_matrix': self.get_confusion_matrix().tolist(),
            'classification_report': self.get_classification_report()
        }

        return metrics


class SegmentationMetrics:
    """Comprehensive segmentation metrics for medical images"""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 num_classes: int, class_names: Optional[List[str]] = None):

        self.y_true = y_true
        self.y_pred = y_pred
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class_{i}' for i in range(num_classes)]

    def dice_coefficient(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        class_id: Optional[int] = None, smooth: float = 1e-7) -> float:
        """Calculate Dice Similarity Coefficient"""
        if class_id is not None:
            y_true = (y_true == class_id).astype(np.float32)
            y_pred = (y_pred == class_id).astype(np.float32)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred)

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return float(dice)

    def jaccard_index(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     class_id: Optional[int] = None, smooth: float = 1e-7) -> float:
        """Calculate Jaccard Index (IoU)"""
        if class_id is not None:
            y_true = (y_true == class_id).astype(np.float32)
            y_pred = (y_pred == class_id).astype(np.float32)

        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection

        jaccard = (intersection + smooth) / (union + smooth)
        return float(jaccard)

    def pixel_accuracy(self) -> float:
        """Calculate pixel-wise accuracy"""
        correct_pixels = np.sum(self.y_true == self.y_pred)
        total_pixels = self.y_true.size
        return float(correct_pixels / total_pixels)

    def hausdorff_distance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Hausdorff distance (simplified version)"""
        # This is a simplified implementation
        # For production use, consider using scipy.spatial.distance
        try:
            from scipy.spatial.distance import directed_hausdorff

            # Get boundary pixels
            true_coords = np.argwhere(y_true > 0)
            pred_coords = np.argwhere(y_pred > 0)

            if len(true_coords) == 0 or len(pred_coords) == 0:
                return float('inf')

            hd1 = directed_hausdorff(true_coords, pred_coords)[0]
            hd2 = directed_hausdorff(pred_coords, true_coords)[0]

            return float(max(hd1, hd2))
        except ImportError:
            # Fallback if scipy is not available
            return 0.0

    def calculate_per_class_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class"""
        per_class_metrics = {}

        for class_id, class_name in enumerate(self.class_names):
            if class_id == 0:  # Skip background class in some cases
                continue

            dice = self.dice_coefficient(self.y_true, self.y_pred, class_id)
            jaccard = self.jaccard_index(self.y_true, self.y_pred, class_id)

            # Calculate precision, recall for this class
            y_true_class = (self.y_true == class_id).astype(int)
            y_pred_class = (self.y_pred == class_id).astype(int)

            tp = np.sum(y_true_class * y_pred_class)
            fp = np.sum((1 - y_true_class) * y_pred_class)
            fn = np.sum(y_true_class * (1 - y_pred_class))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            per_class_metrics[class_name] = {
                'dice_coefficient': dice,
                'jaccard_index': jaccard,
                'precision': float(precision),
                'recall': float(recall)
            }

        return per_class_metrics

    def calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall segmentation metrics"""
        # Mean Dice across all classes (excluding background)
        dice_scores = []
        jaccard_scores = []

        for class_id in range(1, self.num_classes):  # Skip background (class 0)
            dice = self.dice_coefficient(self.y_true, self.y_pred, class_id)
            jaccard = self.jaccard_index(self.y_true, self.y_pred, class_id)

            dice_scores.append(dice)
            jaccard_scores.append(jaccard)

        return {
            'mean_dice': float(np.mean(dice_scores)) if dice_scores else 0.0,
            'mean_jaccard': float(np.mean(jaccard_scores)) if jaccard_scores else 0.0,
            'pixel_accuracy': self.pixel_accuracy()
        }

    def calculate_all_metrics(self) -> Dict:
        """Calculate all segmentation metrics"""
        return {
            'overall_metrics': self.calculate_overall_metrics(),
            'per_class_metrics': self.calculate_per_class_metrics()
        }


class MedicalImageMetrics:
    """Combined metrics class for medical image analysis"""

    @staticmethod
    def sensitivity(tp: int, fn: int) -> float:
        """Calculate sensitivity (recall)"""
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def specificity(tn: int, fp: int) -> float:
        """Calculate specificity"""
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    @staticmethod
    def positive_predictive_value(tp: int, fp: int) -> float:
        """Calculate positive predictive value (precision)"""
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def negative_predictive_value(tn: int, fn: int) -> float:
        """Calculate negative predictive value"""
        return tn / (tn + fn) if (tn + fn) > 0 else 0.0

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                      class_names: List[str], save_path: Optional[str] = None):
        """Plot ROC curves for multi-class classification"""
        plt.figure(figsize=(12, 8))

        if len(class_names) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            auc = roc_auc_score(y_true, y_prob[:, 1])

            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])

                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def main():
    """Test metrics functionality"""
    # Generate sample data for testing
    np.random.seed(42)

    # Classification test
    y_true_clf = np.random.randint(0, 3, 100)
    y_pred_clf = np.random.randint(0, 3, 100)
    y_prob_clf = np.random.rand(100, 3)
    y_prob_clf = y_prob_clf / y_prob_clf.sum(axis=1, keepdims=True)  # Normalize

    clf_metrics = ClassificationMetrics(
        y_true=y_true_clf,
        y_pred=y_pred_clf,
        y_prob=y_prob_clf,
        class_names=['Normal', 'Abnormal_A', 'Abnormal_B']
    )

    print("Classification Metrics:")
    all_metrics = clf_metrics.calculate_all_metrics()
    for key, value in all_metrics['basic_metrics'].items():
        print(f"  {key}: {value:.4f}")

    # Segmentation test
    y_true_seg = np.random.randint(0, 3, (50, 50, 20))
    y_pred_seg = np.random.randint(0, 3, (50, 50, 20))

    seg_metrics = SegmentationMetrics(
        y_true=y_true_seg,
        y_pred=y_pred_seg,
        num_classes=3,
        class_names=['Background', 'Organ', 'Tumor']
    )

    print("
Segmentation Metrics:")
    seg_all_metrics = seg_metrics.calculate_all_metrics()
    for key, value in seg_all_metrics['overall_metrics'].items():
        print(f"  {key}: {value:.4f}")

    print("
Metrics testing completed!")


if __name__ == "__main__":
    main()
