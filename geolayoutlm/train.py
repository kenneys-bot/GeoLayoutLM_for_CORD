import faulthandler
faulthandler.enable()

import os
import torch
from pytorch_lightning import Trainer
# from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning import seed_everything

from lightning_modules.data_modules.vie_data_module import VIEDataModule
from lightning_modules.geolayoutlm_vie_module import GeoLayoutLMVIEModule
from utils import get_callbacks, get_config, get_loggers, get_plugins
import time


CUDA_LAUNCH_BLOCKING=1

def main():
    start = time.perf_counter()

    cfg = get_config()
    print(cfg)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # prevent deadlock with tokenizer
    seed_everything(cfg.seed)

    callbacks = get_callbacks(cfg)
    plugins = get_plugins(cfg)
    loggers = get_loggers(cfg)

    # trainer = Trainer(
    #     accelerator=cfg.train.accelerator,
    #     gpus=torch.cuda.device_count(),
    #     max_epochs=cfg.train.max_epochs,
    #     gradient_clip_val=cfg.train.clip_gradient_value,
    #     gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
    #     callbacks=callbacks,
    #     plugins=plugins,
    #     sync_batchnorm=True,
    #     precision=16 if cfg.train.use_fp16 else 32,
    #     detect_anomaly=False,
    #     replace_sampler_ddp=False,
    #     move_metrics_to_cpu=False,
    #     progress_bar_refresh_rate=0,
    #     check_val_every_n_epoch=cfg.train.val_interval,
    #     logger=loggers,
    #     benchmark=cfg.cudnn_benchmark,
    #     deterministic=cfg.cudnn_deterministic,
    #     limit_val_batches=cfg.val.limit_val_batches,
    # )

    trainer = Trainer(
        accelerator=cfg.train.accelerator,
        devices=torch.cuda.device_count(),
        max_epochs=cfg.train.max_epochs,
        gradient_clip_val=cfg.train.clip_gradient_value,
        gradient_clip_algorithm=cfg.train.clip_gradient_algorithm,
        callbacks=callbacks,
        plugins=plugins,
        sync_batchnorm=True,
        precision=16 if cfg.train.use_fp16 else 32,
        detect_anomaly=False,
        check_val_every_n_epoch=cfg.train.val_interval,
        logger=loggers, 
        benchmark=cfg.cudnn_benchmark,
        deterministic=cfg.cudnn_deterministic,
        limit_val_batches=cfg.val.limit_val_batches,
        log_every_n_steps=10,
    )

    if cfg.model.head in ["vie"]:
        pl_module = GeoLayoutLMVIEModule(cfg)
    else:
        raise ValueError(f"Not supported head {cfg.model.head}")

    # import ipdb;ipdb.set_trace()
    data_module = VIEDataModule(cfg, pl_module.net.tokenizer)

    # torch.set_float32_matmul_precision('medium')

    trainer.fit(pl_module, datamodule=data_module, ckpt_path=None)

    end = time.perf_counter()
    elapsed = end - start
    print("************************************")
    print(f"程序运行时间: {elapsed}s")
    print("************************************")


if __name__ == "__main__":
    main()
