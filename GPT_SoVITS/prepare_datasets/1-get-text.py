# -*- coding: utf-8 -*-

import os

inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
import torch

is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
version = os.environ.get("version", None)
import traceback
import os.path
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tools.my_utils import clean_path

# inp_text=sys.argv[1]
# inp_wav_dir=sys.argv[2]
# exp_name=sys.argv[3]
# i_part=sys.argv[4]
# all_parts=sys.argv[5]
# os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[6]#i_gpu
# opt_dir="/data/docker/liujing04/gpt-vits/fine_tune_dataset/%s"%exp_name
# bert_pretrained_dir="/data/docker/liujing04/bert-vits2/Bert-VITS2-master20231106/bert/chinese-roberta-wwm-ext-large"

from time import time as ttime
import shutil


def my_save(fea, path):  #####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path = "%s%s.pth" % (ttime(), i_part)
    torch.save(fea, tmp_path)
    shutil.move(tmp_path, "%s/%s" % (dir, name))


txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
if os.path.exists(txt_path) == False:
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    if torch.cuda.is_available():
        device = "cuda:0"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    if os.path.exists(bert_pretrained_dir):
        ...
    else:
        raise FileNotFoundError(bert_pretrained_dir)
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)
    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(device)
            res = bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(data, res):
        for name, text, lan in data:
            try:
                name = clean_path(name)
                name = os.path.basename(name)
                print(name)
                phones, word2ph, norm_text = clean_text(text.replace("%", "-").replace("￥", ","), lan, version)
                path_bert = "%s/%s.pt" % (bert_dir, name)
                # ---- 2026-10-25 修正：复用 BERT 特征前必须核对长度 ----
                #
                # 原逻辑是「3-bert/{name}.pt 存在就跳过」。但 BERT 特征是由
                # **同一份文本**的音素展开来的，一旦标注文本被改过，音素个数
                # 就变了，旧的 .pt 会和新的音素序列**长度对不上**。
                #
                # 后果不是报错退出，而是训练在第一个 step 崩在
                # AR/data/dataset.py 的
                #     assert bert_feature.shape[-1] == len(phoneme_ids)
                # 上，而那里离原因十万八千里——用户看到的是
                # "BERT feature dimension (131) ... does not match (136)"，
                # 完全猜不到是「改了标注之后 3-bert 没重算」。
                #
                # 而 3-bert 的特征还会被整个训练集共用：只要有 1 个文件错位，
                # 整个实验就训不出 checkpoint（列表里也不会出现你训练的模型）。
                #
                # 现在改成：读 .len 指纹（没有就读一次 .pt 的形状），
                # 长度对得上才复用，对不上就重算。
                need_bert = True
                if os.path.exists(path_bert):
                    cached_len = None
                    try:
                        with open(path_bert + ".len", "r", encoding="utf-8") as f:
                            cached_len = int(f.read().strip())
                    except (OSError, ValueError):
                        cached_len = None
                    if cached_len is None:
                        # 老数据没有指纹：读一次 .pt 的形状核对（不重算，代价小）
                        try:
                            cached_len = int(torch.load(path_bert, map_location="cpu").shape[-1])
                        except Exception:
                            cached_len = None
                    if cached_len == len(phones):
                        need_bert = False
                    else:
                        print(
                            "BERT 特征长度(%s)与音素序列长度(%d)不一致，重新计算: %s"
                            % (cached_len, len(phones), name)
                        )
                if need_bert and lan == "zh":
                    bert_feature = get_bert_feature(norm_text, word2ph)
                    assert bert_feature.shape[-1] == len(phones)
                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)
                    # 写指纹：下次复用前先比长度
                    try:
                        with open(path_bert + ".len", "w", encoding="utf-8") as f:
                            f.write(str(len(phones)))
                    except OSError:
                        pass
                phones = " ".join(phones)
                # res.append([name,phones])
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())

    todo = []
    res = []
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "KO": "ko",
        "Ko": "ko",
        "ko": "ko",
        "yue": "yue",
        "YUE": "yue",
        "Yue": "yue",
    }
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            # todo.append([name,text,"zh"])
            if language in language_v1_to_language_v2.keys():
                todo.append([wav_name, text, language_v1_to_language_v2.get(language, language)])
            else:
                print(f"\033[33m[Waring] The {language = } of {wav_name} is not supported for training.\033[0m")
        except:
            print(line, traceback.format_exc())

    process(todo, res)
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
