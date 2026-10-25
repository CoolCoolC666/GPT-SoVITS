"""训练前的数据集自检：BERT 特征与音素序列是否对齐。

## 为什么要有这个模块

GPT-SoVITS 的 s1（GPT）训练在 `GPT_SoVITS/AR/data/dataset.py` 里有��句硬断言：

    assert bert_feature.shape[-1] == len(phoneme_ids)

不一致就 AssertionError，**训练在第一个 step 就崩**，而且崩在**任何 checkpoint
落盘之前**——所以「GPT 模型列表里没有我训练的模型」「训练跑完却什么都没有」
这两个现象，根因常常就是这一条断言。

而断言的报错离原因很远：

    AssertionError: The BERT feature dimension (131) of the file 'xxx.wav'
    does not match the length of the phoneme sequence (136).

用户根本猜不到「你改了标注文本，但 3-bert 里的旧特征没被重算」。

所以把校验**提前到训练之前**，用一句能看懂的话告诉用户该做什么。

## 为什么会错位

`GPT_SoVITS/prepare_datasets/1-get-text.py` 里，BERT 特征是**增量复用**的：
`3-bert/{name}.pt` 已存在就跳过重算。而 BERT 特征是由同一份文本的音素展开来的
——标注文本一改，音素个数就变，旧的 .pt 就和新的音素对不上了。

（该脚本已在 2026-10-25 加上长度指纹校验，这里是**兜底**：万一还有别的路径
造成错位，训练前也能拦下来。）
"""

from __future__ import annotations

import os
from typing import List, Tuple


def _read_phoneme_counts(exp_dir: str) -> dict:
    """读 2-name2text.txt，返回 {文件名: 音素个数}。

    合并后的文件每行 4 列：name\\tphones\\tword2ph\\tnorm_text
    音素个数 = 第 2 列按空格切开的数量。
    """
    path = os.path.join(exp_dir, "2-name2text.txt")
    counts = {}
    if not os.path.isfile(path):
        return counts
    with open(path, "r", encoding="utf-8") as f:
        for line in f.read().strip("\n").split("\n"):
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            counts[parts[0]] = len(parts[1].split())
    return counts


def check_bert_alignment(exp_dir: str) -> Tuple[bool, str, List[str]]:
    """校验 3-bert 特征长度与音素长度是否一致。

    Returns:
        (是否对齐, 给用户看的说明, 错配的文件名列表)
    """
    bert_dir = os.path.join(exp_dir, "3-bert")
    counts = _read_phoneme_counts(exp_dir)

    if not counts:
        return True, "（没有 2-name2text.txt，跳过检查）", []
    if not os.path.isdir(bert_dir):
        return True, "（没有 3-bert 目录，跳过检查）", []

    try:
        import torch
    except ImportError:
        return True, "（torch 不可用，跳过检查）", []

    mismatched = []
    for name, n_phones in counts.items():
        pt = os.path.join(bert_dir, f"{name}.pt")
        if not os.path.isfile(pt):
            continue
        try:
            feat = torch.load(pt, map_location="cpu")
            dim = int(feat.shape[-1])
        except Exception:
            continue
        if dim != n_phones:
            mismatched.append(name)

    if not mismatched:
        return True, f"（{len(counts)} 条样本的 BERT 特征与音素长度全部一致）", []

    msg = (
        f"⚠️ 有 {len(mismatched)} 个文件的 BERT 特征与音素序列长度对不上，"
        f"训练会在第一步就崩（且不会产出任何模型）。\n"
        f"原因：标注文本改动后，3-bert 里的旧特征没有被重算。\n"
        f"修法：重跑一次「1A-训练集格式化工具」即可（已自动改为按长度校验后重算）。\n"
        f"涉及文件（前 5 个）：{', '.join(mismatched[:5])}"
    )
    return False, msg, mismatched
