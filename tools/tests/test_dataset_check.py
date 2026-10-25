"""数据集自检（BERT 特征 vs 音素长度对齐）的单元测试。

跑法（在 GPT-SoVITS 项目根）::

    runtime/python.exe tools/tests/test_dataset_check.py

## 为什么值得单独测

这个校验是**训练前的最后一道闸**。它本身如果坏了（永远返回 True），
那训练就还是会崩在 AR/data/dataset.py 那句断言上，而且报错离原因很远。

所以这里造三种情况各测一遍：
1. 全部对齐      → 必须放行
2. 有错配        → 必须拦下，并说清是哪些文件
3. 缺文件/空目录 → 不能误报（否则用户会被无意义的警告挡住）
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from tools.dataset_check import check_bert_alignment  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name}: {detail}")
        print(f"  FAIL  {name}  {detail}")


def build_dataset(base: Path, n: int, wrong: int = 0) -> Path:
    """造一个实验目录：n 个样本，其中 wrong 个的 BERT 维度故意对不上。"""
    exp = base / "exp"
    bert = exp / "3-bert"
    bert.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(n):
        name = f"clip_{i}.wav"
        phones = ["a"] * (10 + i)
        lines.append(f"{name}\t{' '.join(phones)}\t1\t文本{i}")
        dim = len(phones)
        if i < wrong:
            dim = len(phones) + 3  # 故意错位
        # ⚠ 形状必须是 (特征维, 音素数) —— 真实 BERT 特征是
        #   phone_level_feature.T，形状 (1024, n_phones)，
        #   检查器取的是 shape[-1]（= 音素数），别造反了。
        torch.save(torch.zeros(8, dim), bert / f"{name}.pt")

    (exp / "2-name2text.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exp


def main() -> int:
    print("=== 全部对齐 → 必须放行 ===")
    with tempfile.TemporaryDirectory() as td:
        exp = build_dataset(Path(td), n=8)
        ok, msg, bad = check_bert_alignment(str(exp))
        check("对齐时返回 True", ok, msg)
        check("对齐时没有错配列表", bad == [], str(bad))
        check("提示语说明已检查", "一致" in msg, msg)

    print("\n=== 有错配 → 必须拦下 ===")
    with tempfile.TemporaryDirectory() as td:
        exp = build_dataset(Path(td), n=8, wrong=3)
        ok, msg, bad = check_bert_alignment(str(exp))
        check("错配时返回 False", not ok, msg)
        check("列出全部 3 个错配文件", len(bad) == 3, str(bad))
        check("提示语说明了怎么修", "重跑" in msg or "1A" in msg, msg)
        check("提示语点明了原因", "标注文本" in msg, msg)

    print("\n=== 缺文件 / 空目录 → 不能误报 ===")
    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        ok, msg, bad = check_bert_alignment(str(empty))
        check("空目录不误报", ok, msg)
        check("空目录无错配列表", bad == [], str(bad))

    print("\n=== 只有音素表、没有 3-bert → 不误报 ===")
    with tempfile.TemporaryDirectory() as td:
        exp = Path(td) / "exp2"
        exp.mkdir()
        (exp / "2-name2text.txt").write_text("a.wav\tx y z\t1\t文本\n", encoding="utf-8")
        ok, msg, bad = check_bert_alignment(str(exp))
        check("缺 3-bert 不误报", ok, msg)

    print("\n" + "=" * 56)
    print(f"通过 {PASS} 项，失败 {FAIL} 项")
    if FAILURES:
        print("\n失败明细：")
        for f in FAILURES:
            print("  -", f)
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
