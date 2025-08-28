#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多轮根因分析脚本
作者：ChatGLM+OpenAI
说明：
    1. 读取日志 CSV，筛选 anomaly != -1 的行
    2. 加载 ChatGLM-6B（可改为 1B/2B）
    3. 对每条异常做一次多轮分析
    4. 可选进入交互模式继续追问
    5. 结果写入 output CSV
使用方式：
    python analyze_rootcause.py --input logs.csv --output report.csv
    python analyze_rootcause.py --input logs.csv --output report.csv --interactive
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

# ---------- 依赖检查 ----------
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
except ImportError:
    print("请先安装 transformers：pip install transformers sentencepiece tqdm")
    sys.exit(1)


# ---------- 1. 读取 & 过滤 ----------
def load_anomaly_logs(csv_path: str) -> pd.DataFrame:
    """读取 CSV 并筛选异常行（anomaly != -1）"""
    df = pd.read_csv(csv_path)
    if 'anomaly' not in df.columns:
        raise ValueError("CSV 必须包含 'anomaly' 列")
    return df[df['anomaly'] != -1].reset_index(drop=True)


# ---------- 2. ChatGLM 加载 ----------
def load_chatglm(model_name: str = "THUDM/chatglm-6b") -> Tuple[AutoTokenizer, AutoModelForCausalLM, pipeline]:
    """返回 tokenizer、model、pipeline"""
    print(f"🔄 正在加载模型 {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", trust_remote_code=True
    )
    chat_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    print("✅ 模型加载完成")
    return tokenizer, model, chat_pipe


# ---------- 3. 生成多轮对话 ----------
def ask_chatglm(
    pipe: pipeline,
    prompt: str,
    history: List[Dict[str, str]] = None,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    pipe: pipeline 对象
    prompt: 用户输入
    history: 之前的对话列表 [{'role': 'user'/'assistant', 'content': str}]
    返回：回答字符串、更新后的 history
    """
    if history is None:
        history = []

    # 拼接成连贯的上下文
    full_prompt = ""
    for turn in history:
        full_prompt += f"{turn['role']}: {turn['content']}\n"
    full_prompt += f"user: {prompt}\nassistant:"

    # 调用模型
    result = pipe(full_prompt, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=512)
    raw_text = result[0]["generated_text"].strip()

    # 只取 assistant 之后的内容
    reply = raw_text.split("assistant:")[-1].strip()

    # 更新历史
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})

    return reply, history


# ---------- 4. 单条异常分析 ----------
def analyze_single(
    pipe: pipeline,
    row: pd.Series,
    verbose: bool = False,
) -> Tuple[str, List[Dict[str, str]]]:
    """
    对单条异常做一次多轮分析
    返回：分析结果字符串、对话历史
    """
    # 第 1 轮：给出根因与排查思路
    prompt1 = (
        f"以下是一条系统日志，出现了异常：\n"
        f"timestamp: {row['timestamp']}\n"
        f"message: {row['message']}\n"
        f"请先给出这条异常的可能根因（技术层面）以及排查思路。"
    )
    reply1, history = ask_chatglm(pipe, prompt1)

    if verbose:
        print("\n--- 第 1 轮回答 ---")
        print(reply1)

    # 第 2 轮：进一步追问（可自行扩展）
    prompt2 = (
        "你提到可能是硬件或网络导致的超时，请问在排查时需要检查哪些硬件指标？"
    )
    reply2, history = ask_chatglm(pipe, prompt2, history)

    if verbose:
        print("\n--- 第 2 轮回答 ---")
        print(reply2)

    # 第 3 轮：软件层面
    prompt3 = (
        "如果硬件 OK，是否还有软件层面的原因？比如固件版本、驱动、配置？"
    )
    reply3, history = ask_chatglm(pipe, prompt3, history)

    if verbose:
        print("\n--- 第 3 轮回答 ---")
        print(reply3)

    # 合并所有回答
    full_analysis = "\n\n".join([reply1, reply2, reply3])
    return full_analysis, history


# ---------- 5. 主流程 ----------
def main(args):
    # 读取日志
    df = load_anomaly_logs(args.input)
    print(f"📊 共 {len(df)} 条异常待分析")

    # 加载模型
    _, _, chat_pipe = load_chatglm(args.model)

    # 结果列表
    results = []

    # 逐条分析
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="分析进度"):
        analysis, _ = analyze_single(chat_pipe, row, verbose=args.verbose)
        results.append({
            "timestamp": row["timestamp"],
            "message": row["message"],
            "analysis": analysis,
        })

    # 写回 CSV
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n✅ 分析完成，结果已写入 {args.output}")

    # 交互模式
    if args.interactive:
        print("\n🔄 进入交互模式（输入 q 退出）")
        history = []
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in {"q", "quit", "exit"}:
                print("退出交互模式")
                break
            reply, history = ask_chatglm(chat_pipe, user_input, history)
            print(f"ChatGLM: {reply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="多轮根因分析脚本（ChatGLM）"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入日志 CSV 路径（包含 timestamp, message, anomaly 列）",
    )
    parser.add_argument(
        "--output",
        default="analysis_report.csv",
        help="输出分析结果 CSV 路径",
    )
    parser.add_argument(
        "--model",
        default="THUDM/chatglm-6b",
        help="ChatGLM 模型名称（可换成 1B/2B 等）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="分析完后进入交互模式继续追问",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="在分析过程中打印每轮回答",
    )
    args = parser.parse_args()
    main(args)
