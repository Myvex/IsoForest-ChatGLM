# src/anomaly_detector.py
"""
Log anomaly detection with TF‑IDF + IsolationForest.

The original file had a few problems:

1. Missing imports (`Iterable`, `List`, `TfidfVectorizer`).
2. The `detector = LogAnomalyDetector(...)` line was indented inside the
   class definition, so it was executed at *class definition time* and
   became a class attribute – not what we want.
3. `fit_from_logs` used `self.tfidf.fit_transform` unconditionally,
   which would re‑fit the vectoriser every time you called it.
4. `is_fitted` flag was never updated when the vectoriser was fitted.
5. The public API (`fit`, `predict`, `decision_function`, `plot_scores`)
   was fine, but the helper methods for logs were a bit confusing.

Below is a cleaned‑up, fully typed implementation that works with
the rest of the repository.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Iterable, Sequence
from typing import List, Optional

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix


class LogAnomalyDetector:
    """
    通过 TF‑IDF + IsolationForest 对日志文本进行异常检测。

    Parameters
    ----------
    contamination : float, default=0.01
        预期异常比例（0~1）。IsolationForest 会根据该比例决定阈值。
    n_estimators : int, default=100
        随机森林树的数量。
    max_samples : {'auto', int, float}, default='auto'
        每棵树在训练时随机采样的样本数。
    random_state : int, default=42
        随机种子，保证可复现。
    tfidf_params : dict, optional
        传给 `TfidfVectorizer` 的参数，例如 `max_features=5000` 等。
    """

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 100,
        max_samples: str | int | float = "auto",
        random_state: int = 42,
        tfidf_params: Optional[dict] = None,
    ) -> None:
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
        )
        self.contamination = contamination
        self.tfidf = TfidfVectorizer(**(tfidf_params or {}))
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # 直接对 DataFrame 训练/预测（与原代码保持兼容）
    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> None:
        """
        直接使用已处理好的特征矩阵（DataFrame）进行训练。
        """
        self.clf.fit(X)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        直接使用已处理好的特征矩阵（DataFrame）进行预测。
        """
        return self.clf.predict(X)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """
        返回 IsolationForest 的决策函数分数。
        """
        return self.clf.decision_function(X)


    def plot_scores(self, X: pd.DataFrame, y: np.ndarray | None = None) -> None:
        """
        绘制决策分数直方图。若提供 y，则在图中标注异常/正常比例。
        """
        scores = self.decision_function(X)
        plt.figure(figsize=(10, 4))
        plt.hist(scores, bins=50, color="steelblue")
        plt.axvline(0, color="red", linestyle="--")
        plt.title("Isolation Forest decision scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        if y is not None:
            # 计算异常比例
            n_anom = np.sum(y == -1)
            total = len(y)
            plt.text(
                0.95,
                0.95,
                f"异常比例: {n_anom / total:.2%}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=plt.gca().transAxes,
            )
        plt.show()

    # ------------------------------------------------------------------
    # 下面是针对日志文本的专用接口
    # ------------------------------------------------------------------

    locations: Dict[int, Tuple[str, int]]

    def _extract_location(self, log: str) -> Tuple[str, int]:
        """从单条日志中提取 slot_id 与 chip_id（若不存在则返回 None）。"""
        slot_match = re.search(r"slot id:\s*([^\s]+)", log)
        chip_match = re.search(r"chip id\s+(\d+)", log)
        slot = slot_match.group(1) if slot_match else None
        chip = int(chip_match.group(1)) if chip_match else None
        return slot, chip

    def _vectorize(self, logs: Iterable[str]) -> np.ndarray:
        """
        把日志文本转换成 TF‑IDF 特征矩阵（dense）。
        """
        if not self.is_fitted:
            X_sparse = self.tfidf.fit_transform(logs)
        else:
            X_sparse = self.tfidf.transform(logs)
        return X_sparse.toarray()

    

    def fit_from_logs(self, logs: Sequence[str]) -> None:
        """
        先做 TF‑IDF 编码，再训练 IsolationForest。
        同时记录每条日志的位置信息。
        """
        self.locations = {}
        X = []
        for idx, log in enumerate(logs):
            slot, chip = self._extract_location(log)
            self.locations[idx] = (slot, chip)
            X.append(log)
        X = self._vectorize(X)          # 这里会自动 fit/transform
        self.clf.fit(X)
        self.is_fitted = True

    def predict_from_logs(self, logs: Sequence[str]) -> np.ndarray:
        """
        先做 TF‑IDF 编码，再返回预测结果（1 / -1）。
        预测完成后，你可以通过 `self.locations` 直接拿到位置信息。
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model not fitted yet. Call `fit_from_logs` first."
            )
        X = self.tfidf.transform(logs).toarray()
        return self.clf.predict(X)

    # ------------------------------------------------------------------
    # 便捷打印结果
    # ------------------------------------------------------------------
    def print_predictions(self, logs: Sequence[str]) -> None:
        """打印每条日志的异常/正常状态，并显示位置信息。"""
        preds = self.predict_from_logs(logs)
        for i, p in enumerate(preds):
            status = "异常" if p == -1 else "正常"
            slot, chip = self.locations.get(i, (None, None))
            loc_str = f"slot={slot}, chip={chip}" if slot is not None else ""
            print(f"[{status}] {loc_str} {logs[i]}")
    
