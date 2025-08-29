# src/main.py

import argparse
import joblib

import pandas as pd
from feature_extractor import LogFeatureExtractor
from anomaly_detector import LogAnomalyDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Log anomaly detection")
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument("--model", default="model.pkl", help="Path to save/load the model")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 读取日志并提取特征
    # ------------------------------------------------------------------
    extractor = LogFeatureExtractor(tfidf_max_features=500)
    raw_df = extractor.load_logs(args.log_file)
    features = extractor.extract_features(raw_df)

    # ------------------------------------------------------------------
    # 2. 训练 / 预测
    # ------------------------------------------------------------------
    if args.train:
        detector = LogAnomalyDetector(contamination=0.05)
        detector.fit(features)
        joblib.dump(detector, args.model)
        print(f"[INFO] 训练完成，模型已保存到 {args.model}")
        return

    detector = joblib.load(args.model)
    preds = detector.predict(features)
    raw_df["anomaly"] = preds

    # ------------------------------------------------------------------
    # 3. 输出异常结果
    # ------------------------------------------------------------------
    anomalies = raw_df[raw_df["anomaly"] == -1]
    unique_anomalies = anomalies.drop_duplicates(subset=["message", "slot", "chip"])
    print(raw_df.columns.tolist())
    print(f"[INFO] 检测到 {len(anomalies)} 条异常日志")
    anomalies.to_csv("anomalies.csv", index=False)
    df = pd.read_csv('anomalies.csv', dtype=str) 
    group_cols = ['timestamp','message', 'slot', 'chip']    # 分组键
    merged_df = df.drop_duplicates(subset=group_cols, keep='first')
    merged_df.to_csv('anomalies.csv', index=False)


    # 打印异常日志并显示 slot 与 chip
    for _, row in anomalies.iterrows():
        slot = row.get("slot", "N/A")
        chip = row.get("chip", "N/A")
        # print(f"[异常] slot={slot} chip={chip} {row['message']}")


if __name__ == "__main__":
    main()
