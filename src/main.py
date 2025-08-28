# src/main.py
import argparse
import pandas as pd
import joblib
import os

from feature_extractor import LogFeatureExtractor
from anomaly_detector import LogAnomalyDetector

def main(log_file: str, model_path: str = None, train: bool = False):
    extractor = LogFeatureExtractor(tfidf_max_features=500)
    raw_df = extractor.load_logs(log_file)
    features = extractor.extract_features(raw_df)

    if train:
        detector = LogAnomalyDetector(contamination=0.01)
        detector.fit(features)
        joblib.dump(detector, model_path)
        print(f"[INFO] 训练完成，模型已保存到 {model_path}")
    else:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        detector: LogAnomalyDetector = joblib.load(model_path)
        preds = detector.predict(features)
        raw_df['anomaly'] = preds
        # 输出异常日志
        anomalies = raw_df[raw_df['anomaly'] == -1]
        print(f"[INFO] 检测到 {len(anomalies)} 条异常日志")
        anomalies.to_csv('anomalies.csv', index=False)

        # 可视化
        detector.plot_scores(features)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolation Forest 日志异常检测")
    parser.add_argument("log_file", help="待分析的日志文件路径")
    parser.add_argument("--model", default="isolation_forest.pkl", help="模型文件路径")
    parser.add_argument("--train", action="store_true", help="是否训练模型")
    args = parser.parse_args()
    main(args.log_file, args.model, args.train)
