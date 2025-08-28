# src/anomaly_detector.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class LogAnomalyDetector:
    def __init__(self,
                 contamination=0.01,
                 n_estimators=100,
                 max_samples='auto',
                 random_state=42):
        """
        :param contamination: 预期异常比例（0~1）
        """
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )
        self.contamination = contamination

    def fit(self, X: pd.DataFrame):
        self.clf.fit(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回 1（正常） 或 -1（异常）
        """
        return self.clf.predict(X)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        决策函数值，越低越异常
        """
        return self.clf.decision_function(X)

    def plot_scores(self, X: pd.DataFrame, y: np.ndarray = None):
        scores = self.decision_function(X)
        plt.figure(figsize=(10, 4))
        plt.hist(scores, bins=50, color='steelblue')
        plt.axvline(0, color='red', linestyle='--')
        plt.title('Isolation Forest decision scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.show()
