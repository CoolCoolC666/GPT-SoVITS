"""云端 ASR 设置面板的**接线**测试（构建 Gradio 面板并真的调用处理函数）。

跑法（在 GPT-SoVITS 项目根）::

    runtime/python.exe tools/asr/tests/smoke_settings_wiring.py

## 为什么要有这个文件

Gradio 的事件规则是：**处理函数的返回值个数必须与 outputs 列表严格一致**，
多一个少一个都会在用户点击时抛
`Number of output components does not match`。

这个 bug 有个讨厌的特点：**成功路径对、失败路径错**。而失败路径往往
（导入失败、配置缺失）正是用户最需要看到提示的时候，点了却报一个
和「导入失败」毫无关系的错。

所以本测试不靠肉眼看代码，而是：
1. 真的构建出面板，拿到 Gradio 内部记录的 outputs 列表
2. 真的**调用**处理函数（成功路径 + 失败路径都调）
3. 比对返回值个数与 outputs 个数

⚠ 本测试会 monkeypatch `import_from_yilan` 来走失败路径，
   但**不会**真的导入（不写任何文件）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _find_fn(app, api_name: str):
    """按 api_name 找到事件处理函数。

    Gradio 4.x 的事件表是 `app.fns`（dict，值是 BlockFunction），
    **不是** `app.dependencies`（那是配置文件里的形式）。
    BlockFunction 上直接有 .fn / .inputs / .outputs / .targets。

    用 `api_name` 匹配而不是按钮文字：Gradio 自动取函数名做 api_name，
    比在组件里翻 label/value 稳得多（Block 对象的 props 在运行期是空的）。
    """
    for bf in (getattr(app, "fns", {}) or {}).values():
        if getattr(bf, "api_name", None) == api_name:
            return bf
    return None


def main() -> int:
    import gradio as gr

    from tools.asr import cloud_config
    from tools.asr import cloud_settings_ui

    print("=== 构建面板 ===")
    with gr.Blocks() as app:
        cloud_settings_ui.build()
    print(f"   组件 {len(app.blocks)} 个，事件 {len(app.fns)} 个")

    import_bf = _find_fn(app, "_import")
    save_bf = _find_fn(app, "_save")
    refresh_bf = _find_fn(app, "_refresh_status")

    check("找到「从一览成文导入」事件", import_bf is not None)
    check("找到「保存设置」事件", save_bf is not None)
    check("找到「检查配置」事件", refresh_bf is not None)

    if import_bf is None:
        print("\n没有导入事件，后续断言无法进行")
        return 1

    n_out = len(import_bf.outputs)
    print(f"\n=== 导入事件的 outputs 数量: {n_out} ===")

    fn = import_bf.fn
    check("事件带得处理函数", callable(fn), str(fn))

    if callable(fn):
        print("\n=== 成功路径（模拟导入成功，不写任何文件）===")
        original = cloud_config.import_from_yilan

        def _fake_success():
            return True, "模拟导入成功", {
                "engine": "xf_asr",
                "concurrency": 6,
                "xf_asr": {"app_id": "A", "secret_key": "S", "language": "cn"},
                "custom_post": {"endpoint": "https://example.invalid/v1", "api_key": "K"},
            }

        cloud_config.import_from_yilan = _fake_success
        try:
            result_ok = fn()
        except Exception as e:
            result_ok = None
            print(f"   调用异常: {type(e).__name__}: {e}")
        finally:
            cloud_config.import_from_yilan = original

        check("成功路径调用不抛异常", result_ok is not None, "见上方异常")
        if result_ok is not None:
            check(
                f"成功路径返回值个数({len(result_ok)}) == outputs({n_out})",
                len(result_ok) == n_out,
                f"差 {len(result_ok) - n_out}",
            )

        print("\n=== 失败路径（导入失败，用户最常撞到的那条）===")

        def _fake_fail():
            return False, "模拟：未找到一览成文的 config.json", None

        cloud_config.import_from_yilan = _fake_fail
        try:
            result_fail = fn()
        except Exception as e:
            result_fail = None
            print(f"   调用异常: {type(e).__name__}: {e}")
        finally:
            cloud_config.import_from_yilan = original

        check("失败路径调用不抛异常", result_fail is not None, "见上方异常")
        if result_fail is not None:
            check(
                f"失败路径返回值个数({len(result_fail)}) == outputs({n_out})",
                len(result_fail) == n_out,
                f"差 {len(result_fail) - n_out}",
            )
            check("失败路径给出可读提示", "模拟" in str(result_fail[1]), str(result_fail[1])[:60])

    print("\n=== 其他事件的输出数量 ===")
    for label, bf, expect in (
        ("🔄 检查配置", refresh_bf, 1),
        ("💾 保存设置", save_bf, 2),
    ):
        if bf is None:
            continue
        check(f"「{label}」outputs 数量 = {expect}", len(bf.outputs) == expect,
              f"实际 {len(bf.outputs)}")

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
