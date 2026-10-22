def get_models():
    model_size_list = [
        "medium",
        "medium.en",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
    ]
    return model_size_list


# 云端引擎用到的两个键：
#   cloud: True  → 界面「⚙ 设置」面板会自动切到该引擎的参数组
#   engine: "xf_asr" / "custom_post" → 与 tools/asr/cloud_config.py 的配置段对应
#
# 语言选项写的是**各引擎自己的语言代码**（讯飞用 cn/en/ja，custom_post 用
# BCP-47 的 zh/en/ja/ko），不是训练侧的 token——训练 token 的映射在
# tools/asr/cloud_common.py 的 map_lang() 里做。
asr_dict = {
    "Fun-ASR-Nano (31语种+方言, 推荐)": {"lang": ["zh", "en", "ja", "ko", "yue", "auto"], "size": ["large"], "path": "funasr_asr.py", "precision": ["float32"]},
    "SenseVoice (极速, 5语种)": {"lang": ["zh", "en", "ja", "ko", "yue", "auto"], "size": ["large"], "path": "funasr_asr.py", "precision": ["float32"]},
    "达摩 ASR (中文经典)": {"lang": ["zh", "yue"], "size": ["large"], "path": "funasr_asr.py", "precision": ["float32"]},
    "Faster Whisper (多语种)": {
        "lang": ["auto", "en", "ja", "ko"],
        "size": get_models(),
        "path": "fasterwhisper_asr.py",
        "precision": ["float32", "float16", "int8"],
    },
    # ===== 云端引擎（移植自「一览成文」，凭证在 ⚙ 设置里配）=====
    "讯飞听见 (云端·5h/500MB·说话人分离)": {
        "lang": ["cn", "en", "ja", "auto"],
        "size": ["cloud"],
        "path": "xfyun_asr.py",
        "precision": ["float32"],
        "cloud": True,
        "engine": "xf_asr",
    },
    "自定义 POST (云端·MiniMax/DashScope/自建)": {
        "lang": ["zh", "en", "ja", "ko", "auto"],
        "size": ["cloud"],
        "path": "custom_post_asr.py",
        "precision": ["float32"],
        "cloud": True,
        "engine": "custom_post",
    },
}
