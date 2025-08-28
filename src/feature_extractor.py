# src/feature_extractor.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class LogFeatureExtractor:
    """
    适配日志格式：
    [YYYY-MM-DD HH:MM:SS]    MESSAGE
    """
    # 1️⃣ 仅提取时间戳和后面的完整消息
    LOG_PATTERN = re.compile(
        r'\[(?P<timestamp>[^]]+)\]\s+(?P<message>.+)'
    )

    # 如需进一步拆分 MESSAGE（例如提取错误码、十六进制数等），
    # 可以在 parse_line 里再用第二个正则做细粒度解析。

    def __init__(self, tfidf_max_features=500):
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def parse_line(self, line: str):
        m = self.LOG_PATTERN.match(line.strip())
        if m:
            return m.groupdict()
        else:
            return None

    def load_logs(self, filepath: str) -> pd.DataFrame:
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self.parse_line(line)
                if parsed:
                    rows.append(parsed)
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("没有可解析的日志行")
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # 时间戳 → epoch、hour、weekday
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['epoch'] = df['timestamp'].astype(int) / 10**9
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday

        # 由于日志里没有 LEVEL 字段，直接把 message 作为文本特征
        tfidf_matrix = self.tfidf.fit_transform(df['message'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        # 组合特征
        features = pd.concat([df[['epoch', 'hour', 'weekday']].reset_index(drop=True),
                              tfidf_df], axis=1)
        return features
