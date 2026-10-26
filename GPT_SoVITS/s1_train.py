# modified from https://github.com/feng-yufei/shared_debugging_code/blob/main/train_t2s.py
import os

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
import argparse
import logging
import pathlib
import platform
import re
from pathlib import Path

import torch
from AR.data.data_module import Text2SemanticDataModule
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from AR.utils.io import load_yaml_config
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger  # WandbLogger
from pytorch_lightning.strategies import DDPStrategy

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_float32_matmul_precision("high")
from collections import OrderedDict

from AR.utils import get_newest_ckpt
from process_ckpt import my_save


# ============ PyTorch 2.6+ 的 weights_only 兼容 ============
#
# PyTorch 2.6 起，torch.load 的 weights_only 默认值由 False 改成 True。
# Lightning 保存的训练 checkpoint 除了张量，还装着 hparams、optimizer_states、
# lr_schedulers、loops 等一堆 Python 对象，其中有 pathlib.WindowsPath
# （Windows 上尤其常见），在安全模式下会被直接拒掉：
#
#   UnpicklingError: Weights only load failed.
#   ... Unsupported global: GLOBAL pathlib.WindowsPath
#
# 麻烦在于：**这个加载发生在 Lightning 内部**（trainer.fit(ckpt_path=...)），
# 仓库里根本没有可以改的 torch.load 调用——所以不能像别处那样
# 直接传 weights_only=False，只能把对象**事先登记进 torch 的安全名单**。
# 安全名单是进程级的，torch.load 每次调用都会查它，事先登记即可。
#
# 上游 RVC-Boss/GPT-SoVITS 目前没处理这个问题（它自己的 Windows 依赖
# 钉的就是 torch==2.7.1），所以每个 torch>=2.6 的用户一恢复 S1 训练就会撞上。

_UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global:\s*GLOBAL\s+([\w\.]+)")


def allow_ckpt_globals() -> None:
    """把 Lightning checkpoint 里已知的对象登记进 torch 安全名单。"""
    add_safe = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add_safe is None:
        return  # torch 版本太老，没有这个 API，本来也不受影响
    safe = [pathlib.WindowsPath, pathlib.PosixPath]
    try:
        add_safe(safe)
    except Exception as e:
        print("登记 torch 安全名单失败（不影响本次之外的流程）:", e)


def register_global_from_error(err: BaseException) -> bool:
    """从 UnpicklingError 里解析出被拒的全局名，登记进安全名单。

    成功返回 True（调用方应当重试）；解析不出来返回 False。
    """
    m = _UNSUPPORTED_GLOBAL_RE.search(str(err))
    if not m:
        return False
    gname = m.group(1)
    mod_path, _, attr = gname.rpartition(".")
    if not mod_path:
        return False
    try:
        mod = __import__(mod_path, fromlist=[attr])
        cls = getattr(mod, attr)
    except Exception:
        return False
    add_safe = getattr(getattr(torch, "serialization", None), "add_safe_globals", None)
    if add_safe is None:
        return False
    add_safe([cls])
    print("已把 %s 加入 torch 安全名单" % gname)
    return True



class my_model_ckpt(ModelCheckpoint):
    def __init__(
        self,
        config,
        if_save_latest,
        if_save_every_weights,
        half_weights_save_dir,
        exp_name,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_save_latest = if_save_latest
        self.if_save_every_weights = if_save_every_weights
        self.half_weights_save_dir = half_weights_save_dir
        self.exp_name = exp_name
        self.config = config

    def on_train_epoch_end(self, trainer, pl_module):
        # if not self._should_skip_saving_checkpoint(trainer) and self._should_save_on_train_epoch_end(trainer):
        if self._should_save_on_train_epoch_end(trainer):
            monitor_candidates = self._monitor_candidates(trainer)
            if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
                if (
                    self.if_save_latest == True
                ):  ####如果设置只保存最后一个ckpt，在保存下一个ckpt后要清理掉之前的所有ckpt
                    to_clean = list(os.listdir(self.dirpath))
                self._save_topk_checkpoint(trainer, monitor_candidates)
                if self.if_save_latest == True:
                    for name in to_clean:
                        try:
                            os.remove("%s/%s" % (self.dirpath, name))
                        except:
                            pass
                if self.if_save_every_weights == True:
                    to_save_od = OrderedDict()
                    to_save_od["weight"] = OrderedDict()
                    dictt = trainer.strategy._lightning_module.state_dict()
                    for key in dictt:
                        to_save_od["weight"][key] = dictt[key].half()
                    to_save_od["config"] = self.config
                    to_save_od["info"] = "GPT-e%s" % (trainer.current_epoch + 1)
                    # torch.save(
                    # print(os.environ)
                    if os.environ.get("LOCAL_RANK", "0") == "0":
                        my_save(
                            to_save_od,
                            "%s/%s-e%s.ckpt"
                            % (
                                self.half_weights_save_dir,
                                self.exp_name,
                                trainer.current_epoch + 1,
                            ),
                        )
            self._save_last_checkpoint(trainer, monitor_candidates)


def main(args):
    config = load_yaml_config(args.config_file)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = output_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(config["train"]["seed"], workers=True)
    ckpt_callback: ModelCheckpoint = my_model_ckpt(
        config=config,
        if_save_latest=config["train"]["if_save_latest"],
        if_save_every_weights=config["train"]["if_save_every_weights"],
        half_weights_save_dir=config["train"]["half_weights_save_dir"],
        exp_name=config["train"]["exp_name"],
        save_top_k=-1,
        monitor="top_3_acc",
        mode="max",
        save_on_train_epoch_end=True,
        every_n_epochs=config["train"]["save_every_n_epoch"],
        dirpath=ckpt_dir,
    )
    logger = TensorBoardLogger(name=output_dir.stem, save_dir=output_dir)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["USE_LIBUV"] = "0"

    def build_trainer() -> Trainer:
        return Trainer(
            max_epochs=config["train"]["epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            # val_check_interval=9999999999999999999999,###不要验证
            # check_val_every_n_epoch=None,
            limit_val_batches=0,
            devices=-1 if torch.cuda.is_available() else 1,
            benchmark=False,
            fast_dev_run=False,
            strategy=DDPStrategy(process_group_backend="nccl" if platform.system() != "Windows" else "gloo")
            if torch.cuda.is_available()
            else "auto",
            precision=config["train"]["precision"],
            logger=logger,
            num_sanity_val_steps=0,
            callbacks=[ckpt_callback],
            use_distributed_sampler=False,  # 非常简单的修改，但解决了采用自定义的 bucket_sampler 下训练步数不一致的问题！
        )

    try:
        # 使用正则表达式匹配文件名中的数字部分，并按数字大小进行排序
        newest_ckpt_name = get_newest_ckpt(os.listdir(ckpt_dir))
        ckpt_path = ckpt_dir / newest_ckpt_name
    except Exception:
        ckpt_path = None
    print("ckpt_path:", ckpt_path)

    # 恢复训练前先放行 checkpoint 里会用到的对象（见文件头的说明）
    allow_ckpt_globals()

    # 每次尝试都重建 trainer / model：checkpoint 加载失败时 trainer 可能已经
    # 处在半初始化状态，复用同一个对象重试不可靠。
    for attempt in range(3):
        trainer = build_trainer()
        model: Text2SemanticLightningModule = Text2SemanticLightningModule(config, output_dir)
        data_module: Text2SemanticDataModule = Text2SemanticDataModule(
            config,
            train_semantic_path=config["train_semantic_path"],
            train_phoneme_path=config["train_phoneme_path"],
            # dev_semantic_path=args.dev_semantic_path,
            # dev_phoneme_path=args.dev_phoneme_path
        )
        try:
            trainer.fit(model, data_module, ckpt_path=ckpt_path)
            return
        except Exception as e:
            # 只有「checkpoint 里有新的不安全全局」才值得重试；
            # 其它异常（OOM、显存不足等）原样抛出，别掩盖真因。
            if ckpt_path is None or not register_global_from_error(e):
                raise
            print("恢复 checkpoint 时遇到新的不安全全局，已登记，重试…（第 %d 次）" % (attempt + 1))
    raise RuntimeError("恢复训练重试多次仍失败，请检查上面的报错")


# srun --gpus-per-node=1 --ntasks-per-node=1 python train.py --path-to-configuration configurations/default.yaml
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        type=str,
        default="configs/s1longer.yaml",
        help="path of config file",
    )
    # args for dataset
    # parser.add_argument('--train_semantic_path',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/6-name2semantic.tsv')
    # parser.add_argument('--train_phoneme_path', type=str, default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/2-name2text.txt')

    # parser.add_argument('--dev_semantic_path', type=str, default='dump_mix/semantic_dev.tsv')
    # parser.add_argument('--dev_phoneme_path', type=str, default='dump_mix/phoneme_dev.npy')
    # parser.add_argument('--output_dir',type=str,default='/data/docker/liujing04/gpt-vits/fine_tune_dataset/xuangou/logs_s1',help='directory to save the results')
    # parser.add_argument('--output_dir',type=str,default='/liujing04/gpt_logs/s1/xuangou_ft',help='directory to save the results')

    args = parser.parse_args()
    logging.info(str(args))
    main(args)
