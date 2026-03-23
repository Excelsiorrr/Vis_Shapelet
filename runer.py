import sys
from omegaconf import OmegaConf

from exp_saliency_general import ShapeXPipline#这是核心类（训练、测试、解释性分析都在里面）
from txai.synth_data.synth_data_base import SynthTrainDataset
import vis.vis_ecg as vis_ecg


#读取配置，然后根据配置去调度核心流水线 (ShapeXPipline) 来执行 训练 (Train) 或 测试/解释性分析 (Test/Saliency)。

SHAPEX_BETA_ROOT = "D:/shapelet/ShapeX"

#一个普通 dict 变成 OmegaConf 配置对象
def build_config(dataset_name: str, num_classes: int = 4, seq_len: int = 500, proto_len: int = 30):
    """
    Construct a config object compatible with exp_saliency_general.get_args.
    The dataset-specific defaults can be overridden by CLI flags that
    exp_saliency_general.get_args will parse from sys.argv.
    """
    # Each dataset must be an attribute on the config with (num_classes, seq_len, proto_len)
    cfg = {
        "base": {"root_dir": SHAPEX_BETA_ROOT},#项目根目录在哪（读 configs、datasets、checkpoints 都可能用它）
        "dataset": {"name": dataset_name, "meta_dataset": "default"},#用于区分不同数据集族/模式（比如 default / ecg / ucr 等）
        dataset_name: {
            "num_classes": num_classes,
            "seq_len": seq_len,
            "proto_len": proto_len,
#             这个是关键：用 dataset_name 当作“配置分组名”

# 意思是：每个数据集有自己一套参数：num_classes / seq_len / proto_len
        },
    }
    return OmegaConf.create(cfg)

#从 configs/run_configs.yaml 中读取 defaults 部分。
def main():
    # Load run params from YAML (only three keys)
    run_cfg = OmegaConf.load(f"{SHAPEX_BETA_ROOT}/configs/run_configs.yaml").defaults#它从 configs/run_configs.yaml 里读取 defaults 段，且只关心三个键：
    datasets = str(run_cfg.datasets)
    do_train = bool(run_cfg.train)
    do_test = bool(run_cfg.test)

    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    #ShapeXPipline 里面大概率会调用一个命令行参数解析器（比如 argparse），它会从 sys.argv 读取参数。
    for ds in dataset_list:
        print(f"===== Running dataset: {ds} =====")
        config = build_config(ds)
        # prevent downstream arg parse from reading CLI
        saved_argv = sys.argv
        sys.argv = [sys.argv[0]]
        pipeline = ShapeXPipline(config)
        sys.argv = saved_argv

        # Train
        if do_train:
            # get_args() inside pipeline will read is_training from CLI; ensure it is set
            # If not provided by user, training flag here implies is_training=1
            # Users can still override via CLI (e.g., --is_training 0)
            pipeline.args.is_training = 1
            pipeline.train_shapex()#训练入口函数
            # After training, render visualizations (save only)
            try:
                vis_ecg.main(show=False, pipeline=pipeline)
            except Exception as e:
                print(f"[WARN] vis_ecg failed: {e}")

        # Evaluate
        if do_test:
            pipeline.args.saliency = True#“这次要做 saliency / 解释性”
            pipeline.eval_shapex()


if __name__ == "__main__":
    main()


