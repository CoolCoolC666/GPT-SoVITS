"""「0c-语音识别工具」里的云端 ASR 设置面板。

## 为什么单独一个文件

`webui.py` 是上游同步维护的大文件。这里放进去的任何东西都会增加将来
跟进上游更新的冲突面，所以整个面板收在本文件，webui.py 里只有一次调用。

## 面板里有什么

- 当前就绪状态（**离线**检查，不发任何请求）
- 讯飞听见 / 自定义 POST 两个引擎的凭证与参数
- 「从一览成文导入」按钮：把一览成文 config.json 里的 ASR 段拷过来
- 并发数：云端引擎按文件并发，靠它提速

## 两条自律

1. **不主动探测远端**。「检查配置」只读本地文件判断「齐不齐」，
   不去 ping 讯飞/自建网关。真正的连通性由用户点「开启语音识别」时验证——
   探测动作本身有成本（额度/风控），不该被当健康检查反复触发。
2. **密钥默认遮挡**。密码框 + 「显示密钥」勾选，页面截图/投屏时不会裸奔。
"""

from __future__ import annotations

import gradio as gr

from . import cloud_config

# 讯飞领域（pd）选项
PD_DOMAINS = {
    "": "通用（不设领域）",
    "edu": "教育（课程/课堂实录）",
    "court": "法律",
    "finance": "金融",
    "medical": "医疗",
    "tech": "科技",
    "sport": "体育",
    "gov": "政府",
    "game": "游戏",
    "ecom": "电商",
    "car": "汽车",
}

API_STYLES = {
    "openai_compat": "OpenAI 兼容（上传文件本体）",
    "dashscope_async": "DashScope 异步（提交→轮询→下载）",
}


def _status_text() -> str:
    state = cloud_config.readiness()
    ok = lambda flag: "✅" if flag else "⚠️"  # noqa: E731
    xf, cp = state["xf_asr"], state["custom_post"]
    return (
        f"{ok(xf['ready'])} **讯飞听见**：{xf['detail']}\n\n"
        f"{ok(cp['ready'])} **自定义 POST**：{cp['detail']}\n\n"
        f"**默认引擎**：{state['engine']}　**并发数**：{state['concurrency']}\n\n"
        f"<details><summary>配置文件位置</summary>\n\n"
        f"- 本地：`{state['config_path']}`"
        f"{'（已存在）' if state['config_exists'] else '（尚未创建）'}\n"
        f"- 一览成文（只读回落）：`{state['yilan_config']}`\n\n"
        f"密钥只存在本机这个文件里，**已加入 .gitignore**，不会进版本库。\n\n"
        f"</details>"
    )


def build() -> dict:
    """在当前 Gradio 上下文里建出设置面板，返回组件字典。"""
    cfg = cloud_config.load()
    xf = cfg.get("xf_asr") or {}
    cp = cfg.get("custom_post") or {}

    with gr.Accordion(label="⚙ 设置（云端 ASR 引擎）", open=False):
        status = gr.Markdown(value=_status_text())
        message = gr.Markdown(value="")

        with gr.Row():
            show_secret = gr.Checkbox(value=False, label="👁 显示密钥（页面截图时注意）")
            refresh_btn = gr.Button(value="🔄 检查配置", variant="secondary", size="sm")
            import_btn = gr.Button(value="📥 从一览成文导入", variant="secondary", size="sm")
            save_btn = gr.Button(value="💾 保存设置", variant="primary", size="sm")

        with gr.Row():
            engine = gr.Dropdown(
                choices=["xf_asr", "custom_post"],
                value=cfg.get("engine", "xf_asr"),
                label="默认云端引擎",
                scale=2,
            )
            concurrency = gr.Number(
                value=int(cfg.get("concurrency", 4) or 4),
                label="并发文件数（云端引擎按文件并发）",
                precision=0,
                minimum=1,
                maximum=16,
                scale=3,
            )

        with gr.Group(visible=(cfg.get("engine", "xf_asr") == "xf_asr")) as xf_group:
            gr.Markdown("**讯飞听见**（长语音转写：单文件 ≤500MB / ≤5 小时，服务端自己处理长音频，不切段）")
            with gr.Row():
                xf_app_id = gr.Textbox(value=str(xf.get("app_id") or ""), label="APPID", scale=2)
                xf_secret = gr.Textbox(
                    value=str(xf.get("secret_key") or ""), label="SecretKey", type="password", scale=2
                )
            with gr.Row():
                xf_lang = gr.Dropdown(
                    choices=["cn", "en", "ja", "auto"], value=str(xf.get("language") or "cn"), label="语言"
                )
                xf_domain = gr.Dropdown(
                    choices=list(PD_DOMAINS.keys()),
                    value=str(xf.get("pd_domain") or ""),
                    label="垂直领域（专用声学+语言模型）",
                )
                xf_role_num = gr.Number(
                    value=int(xf.get("role_num", 2) or 2), label="说话人数（0=自动盲分）", precision=0
                )
            with gr.Row():
                xf_role = gr.Checkbox(
                    value=bool(xf.get("role_separation")), label="说话人分离（需账号开通角色分离权限）"
                )
                xf_colloq = gr.Checkbox(
                    value=bool(xf.get("colloquial_proc")), label="口语规整（去「嗯啊呃」与口癖）"
                )
                xf_stamp = gr.Checkbox(value=bool(xf.get("timestamps", True)), label="输出时间戳")
                xf_mock = gr.Checkbox(value=bool(xf.get("mock")), label="Mock 模式（不调真实 API）")
            xf_wait = gr.Number(
                value=int(xf.get("max_wait_seconds", 600) or 600), label="轮询超时（秒）", precision=0
            )

        with gr.Group(visible=(cfg.get("engine", "xf_asr") == "custom_post")) as cp_group:
            gr.Markdown(
                "**自定义 POST**（同一个框接两类协议完全不同的服务，"
                "风格填错会被服务端直接断连，详见 https://github.com/CoolCoolC666/video-to-article）"
            )
            with gr.Row():
                cp_endpoint = gr.Textbox(
                    value=str(cp.get("endpoint") or ""),
                    label="接口地址（OpenAI 兼容风格只填到 /v1，路径由程序拼）",
                    scale=3,
                )
                cp_style = gr.Dropdown(
                    choices=list(API_STYLES.keys()), value=str(cp.get("api_style") or "openai_compat"),
                    label="API 风格",
                    scale=2,
                )
            with gr.Row():
                cp_api_key = gr.Textbox(
                    value=str(cp.get("api_key") or ""), label="API Key", type="password", scale=2
                )
                cp_model = gr.Textbox(value=str(cp.get("model") or ""), label="模型名（DashScope 必填）", scale=2)
            with gr.Row():
                cp_headers = gr.Textbox(
                    value=str(cp.get("headers_file") or ""), label="附加请求头文件路径（留空用默认）", scale=3
                )
                cp_lang = gr.Dropdown(
                    choices=["zh", "en", "ja", "ko", "auto"],
                    value=str(cp.get("language") or "auto"),
                    label="语言（留空=自动检测）",
                    scale=2,
                )
            with gr.Row():
                cp_role = gr.Checkbox(value=bool(cp.get("role_separation")), label="说话人分离")
                cp_spk = gr.Number(value=int(cp.get("speaker_count", 2) or 2), label="说话人数", precision=0)
                cp_stamp = gr.Checkbox(value=bool(cp.get("timestamps", True)), label="输出时间戳")
                cp_mock = gr.Checkbox(value=bool(cp.get("mock")), label="Mock 模式（不调真实 API）")
            cp_wait = gr.Number(
                value=int(cp.get("max_wait_seconds", 600) or 600), label="轮询超时（秒）", precision=0
            )

    # ---------------- 事件 ----------------

    def _toggle_secret(visible):
        return {"__type": "update", "type": "text" if visible else "password"}

    def _switch_engine(key):
        return {
            "__type": "update",
            "visible": key == "xf_asr",
        }, {
            "__type": "update",
            "visible": key == "custom_post",
        }

    def _save(
        engine_v, concurrency_v, app_id, secret, lang, domain, role, role_num, colloq, stamp, mock, wait,
        endpoint, style, api_key, model, headers, cp_lang, cp_role, cp_spk, cp_stamp, cp_mock, cp_wait,
    ):
        payload = {
            "engine": engine_v,
            "concurrency": int(concurrency_v or 4),
            "xf_asr": {
                "app_id": str(app_id or "").strip(),
                "secret_key": str(secret or "").strip(),
                "language": lang or "cn",
                "pd_domain": domain or "",
                "role_separation": bool(role),
                "role_num": int(role_num or 2),
                "colloquial_proc": bool(colloq),
                "timestamps": bool(stamp),
                "mock": bool(mock),
                "max_wait_seconds": int(wait or 600),
            },
            "custom_post": {
                "endpoint": str(endpoint or "").strip(),
                "api_style": style or "openai_compat",
                "api_key": str(api_key or "").strip(),
                "model": str(model or "").strip(),
                "headers_file": str(headers or "").strip(),
                "language": cp_lang or "",
                "role_separation": bool(cp_role),
                "speaker_count": int(cp_spk or 2),
                "timestamps": bool(cp_stamp),
                "mock": bool(cp_mock),
                "max_wait_seconds": int(cp_wait or 600),
            },
        }
        path = cloud_config.save(payload)
        return _status_text(), f"已保存到 {path}"

    def _import():
        """从一览成文导入。**成功与失败都返回同样数量的输出**——
        Gradio 的 outputs 数量必须与返回值数量严格一致，少一个就会在点击时报错。"""
        ok, msg, imported = cloud_config.import_from_yilan()

        if not ok or not imported:
            # 失败：只更新状态与提示，字段一律不动
            #
            # ⚠ 返回值个数必须与 outputs 列表**严格一致**（2 + 23 = 25）：
            #   2  = 状态 + 提示
            #   23 = 引擎 + 并发 + 讯飞 10 项 + 自定义 POST 11 项
            # 多一个少一个，Gradio 在用户点击时抛
            # "Number of output components does not match"——
            # 而这条路径恰恰是「导入失败」时才会走到的，最容易漏测。
            return (
                _status_text(),
                f"❌ {msg}",
                *(gr.update() for _ in range(23)),
            )

        xf2 = imported.get("xf_asr") or {}
        cp2 = imported.get("custom_post") or {}
        return (
            _status_text(),
            f"✅ {msg}（已拷贝到本地配置；之后一览成文那边改动不会自动同步）",
            gr.update(value=imported.get("engine", "xf_asr")),
            gr.update(value=int(imported.get("concurrency", 4) or 4)),
            gr.update(value=str(xf2.get("app_id") or "")),
            gr.update(value=str(xf2.get("secret_key") or "")),
            gr.update(value=str(xf2.get("language") or "cn")),
            gr.update(value=str(xf2.get("pd_domain") or "")),
            gr.update(value=bool(xf2.get("role_separation"))),
            gr.update(value=int(xf2.get("role_num", 2) or 2)),
            gr.update(value=bool(xf2.get("colloquial_proc"))),
            gr.update(value=bool(xf2.get("timestamps", True))),
            gr.update(value=bool(xf2.get("mock"))),
            gr.update(value=int(xf2.get("max_wait_seconds", 600) or 600)),
            gr.update(value=str(cp2.get("endpoint") or "")),
            gr.update(value=str(cp2.get("api_style") or "openai_compat")),
            gr.update(value=str(cp2.get("api_key") or "")),
            gr.update(value=str(cp2.get("model") or "")),
            gr.update(value=str(cp2.get("headers_file") or "")),
            gr.update(value=str(cp2.get("language") or "auto")),
            gr.update(value=bool(cp2.get("role_separation"))),
            gr.update(value=int(cp2.get("speaker_count", 2) or 2)),
            gr.update(value=bool(cp2.get("timestamps", True))),
            gr.update(value=bool(cp2.get("mock"))),
            gr.update(value=int(cp2.get("max_wait_seconds", 600) or 600)),
        )

    def _refresh_status():
        """只读本地状态，不发任何网络请求。"""
        return _status_text()

    show_secret.change(_toggle_secret, inputs=[show_secret], outputs=[xf_secret, cp_api_key])
    engine.change(_switch_engine, inputs=[engine], outputs=[xf_group, cp_group])
    refresh_btn.click(_refresh_status, outputs=status)
    save_btn.click(
        _save,
        inputs=[
            engine, concurrency,
            xf_app_id, xf_secret, xf_lang, xf_domain, xf_role, xf_role_num, xf_colloq, xf_stamp, xf_mock, xf_wait,
            cp_endpoint, cp_style, cp_api_key, cp_model, cp_headers, cp_lang, cp_role, cp_spk, cp_stamp, cp_mock, cp_wait,
        ],
        outputs=[status, message],
    )
    import_btn.click(
        _import,
        outputs=[
            status, message,
            engine, concurrency,
            xf_app_id, xf_secret, xf_lang, xf_domain, xf_role, xf_role_num, xf_colloq, xf_stamp, xf_mock, xf_wait,
            cp_endpoint, cp_style, cp_api_key, cp_model, cp_headers, cp_lang, cp_role, cp_spk, cp_stamp, cp_mock, cp_wait,
        ],
    )

    return {
        "status": status,
        "engine": engine,
        "concurrency": concurrency,
    }
