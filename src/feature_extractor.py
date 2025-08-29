# src/feature_extractor.py
"""
日志特征提取器

适配日志格式：
[YYYY-MM-DD HH:MM:SS]    MESSAGE

新增功能：
- 维护 slot / chip 的上下文（如果日志中出现 “slot id:” / “chip id” 行）
- 将 slot / chip 编码为数值特征
"""

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

    def __init__(self, tfidf_max_features: int = 500):
        """
        Parameters
        ----------
        tfidf_max_features : int
            TfidfVectorizer 的 max_features 参数，控制词向量维度
        """
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def parse_line(self, line: str):
        """
        解析单行日志，返回字典
        """
        m = self.LOG_PATTERN.match(line.strip())
        if m:
            return m.groupdict()
        else:
            return None

    def load_logs(self, filepath: str) -> pd.DataFrame:
        """
        读取日志文件，返回 DataFrame

        该实现会维护 slot / chip 的上下文信息：
        - 当遇到以 "slot id:" 开头的行时，更新当前 slot
        - 当遇到以 "chip id" 开头的行时，更新当前 chip
        - 其余行会将当前 slot / chip 作为上下文字段保留
        """
        rows = []
        cur_slot = None
        cur_chip = None

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self.parse_line(line)
                if not parsed:
                    continue

                # 先检查是否是 slot / chip 记录
                if parsed['message'].startswith('slot id:'):
                    cur_slot = parsed['message'].split(':', 1)[1].strip()
                    continue

                if parsed['message'].startswith('chip id'):
                    # 可能是 "chip id 1" 或 "chip id 1 "
                    cur_chip = parsed['message'].split(' ', 2)[-1].strip()
                    continue

                # 其它行视为日志内容，保留当前上下文
                rows.append({
                    'timestamp': parsed['timestamp'],
                    'message': parsed['message'],
                    'slot': cur_slot,
                    'chip': cur_chip
                })

        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("没有可解析的日志行")
        return df

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        从日志 DataFrame 提取特征

        特征包括：
        - epoch, hour, weekday（时间特征）
        - slot_code, chip_code（如果存在）
        - tfidf 向量（文本特征）
        """
        # 时间戳 → epoch、hour、weekday
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # 以秒为单位的 epoch 时间（浮点）
        df['epoch'] = df['timestamp'].astype('int64') / 10**9
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday

        # 由于日志里没有 LEVEL 字段，直接把 message 作为文本特征
        tfidf_matrix = self.tfidf.fit_transform(df['message'])
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        )

        # 编码 slot 与 chip（如果有的话）
        if 'slot' in df.columns:
            slot_codes, _ = pd.factorize(df['slot'])
            df['slot_code'] = slot_codes
        if 'chip' in df.columns:
            chip_codes, _ = pd.factorize(df['chip'])
            df['chip_code'] = chip_codes

        # 组合特征
        feature_columns = ['epoch', 'hour', 'weekday']
        if 'slot_code' in df.columns:
            feature_columns.append('slot_code')
        if 'chip_code' in df.columns:
            feature_columns.append('chip_code')

        features = pd.concat(
            [df[feature_columns].reset_index(drop=True), tfidf_df],
            axis=1
        )
        return features
