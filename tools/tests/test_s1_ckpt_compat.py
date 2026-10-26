"""验证 PyTorch 2.6+ 的 weights_only 兼容修复。

跑法（在 GPT-SoVITS 项目根）::

    runtime/python.exe tools/tests/test_s1_ckpt_compat.py

## 测什么

1. **问题复现**：不带白名单加载含 pathlib.WindowsPath 的 checkpoint → 失败
2. **修复生效**：调 allow_ckpt_globals() 后同一个 checkpoint → 成功
3. **自动兜底**：模拟一个「报错里带未知全局名」的情况，
   register_global_from_error() 能解析出来并登记
4. **不掩盖真因**：非 pickle 类异常不会被当成「需要重试」

## 为什么第 1 条要留着

不证明「原来真的会炸」，就无法证明修复真的修的是那个问题——
这是本仓库一贯的做法：修复必须有能失败的旧行为做对照。
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "GPT_SoVITS") not in sys.path:
    sys.path.insert(0, str(ROOT / "GPT_SoVITS"))

import torch  # noqa: E402

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


def main() -> int:
    import s1_train

    print(f"torch {torch.__version__}")
    print()

    print("=== 1. 造一个含 WindowsPath 的 checkpoint ===")
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "epoch=0-step=0.ckpt"
        payload = {
            "epoch": 0,
            "global_step": 0,
            "state_dict": {"w": torch.zeros(2, 2)},
            "hyper_parameters": {"config": {"output_dir": pathlib.Path(td) / "out"}},
        }
        torch.save(payload, ckpt)
        print(f"   已生成 {ckpt.name}（含 pathlib.WindowsPath）")

        print("\n=== 2. 不开白名单 → 应当失败（这正是用户的报错）===")
        # 先把白名单清掉，确保测的是「没有白名单」的状态
        try:
            torch.serialization.add_safe_globals([])
        except Exception:
            pass
        # 移除之前可能被登记的 WindowsPath
        try:
            torch.serialization._weights_only_unsafe_globals.discard(pathlib.WindowsPath)
        except Exception:
            pass
        failed = False
        try:
            torch.load(str(ckpt), map_location="cpu", weights_only=True)
        except Exception as e:
            failed = True
            print(f"   如期失败: {type(e).__name__}: {str(e)[:110]}")
        check("未登记时 weights_only 加载失败", failed, "本机上没复现出失败，测试环境可能不同")

        print("\n=== 3. 调 allow_ckpt_globals() → 应当成功 ===")
        s1_train.allow_ckpt_globals()
        loaded = None
        try:
            loaded = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            ok = True
        except Exception as e:
            ok = False
            print(f"   仍失败: {e}")
        check("登记白名单后加载成功", ok)
        if loaded is not None:
            check("内容完整", loaded.get("epoch") == 0 and "state_dict" in loaded, str(list(loaded)))

    print("\n=== 4. 自动兜底：从报错里解析并登记未知全局 ===")
    # ⚠ 用 argparse.Namespace 做例子：它是 Lightning checkpoint 里真实会出现的
    #   类型，而且**不在** torch 的默认白名单里。
    #   （别用 collections.OrderedDict —— 那个默认就允许，测不出东西）
    import argparse

    try:
        torch.serialization._weights_only_unsafe_globals.discard(argparse.Namespace)
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as td:
        ckpt2 = Path(td) / "c.ckpt"
        torch.save({"ns": argparse.Namespace(a=1)}, ckpt2)
        failed2 = False
        try:
            torch.load(str(ckpt2), map_location="cpu", weights_only=True)
        except Exception:
            failed2 = True
        check("未登记 argparse.Namespace 时失败", failed2)
        s1_train.register_global_from_error(
            Exception("Unsupported global: GLOBAL argparse.Namespace")
        )
        ok2 = True
        try:
            torch.load(str(ckpt2), map_location="cpu", weights_only=True)
        except Exception as e:
            ok2 = False
            print("   仍失败:", e)
        check("登记后 argparse.Namespace 也能加载", ok2)

    print("\n=== 5. 非 pickle 类异常不应被当成可重试 ===")
    check("OOM 类异常返回 False",
          s1_train.register_global_from_error(RuntimeError("CUDA out of memory")) is False)
    check("普通 ValueError 返回 False",
          s1_train.register_global_from_error(ValueError("随便什么错")) is False)

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
