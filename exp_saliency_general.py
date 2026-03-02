"""ShapeX experiment pipeline: training classification explainer and saliency evaluation.

This module defines the pipeline to train the ProtoPTST explainer and evaluate
saliency explanations against ground truth for supported datasets.
"""#将数据、模型（解释器）、优化器和评估指标组装在一起，执行具体的 训练 和 解释性评估 任务
import os
import time
import numpy as np
import torch
from torch import nn, optim
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import argparse
from torch.utils.data import DataLoader
import sys
from omegaconf import OmegaConf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from txai.models.encoders.transformer_simple import TransformerMVTS#解释评估时需要一个“原分类器”来算分；有些数据集会从 checkpoint 加载训练好的分类模型
from txai.models.encoders.simple import CNN, LSTM
from txai.utils.evaluation import ground_truth_xai_eval#把你算出来的 saliency score 跟 gt_exps 做对比，输出指标
from utils_training import EarlyStopping#早停（验证集指标不提升就停）
from get_data import get_saliency_data#统一把数据读成训练/验证/测试集，以及（做解释评估时）对应的 ground-truth explanation
from shapelet_encoder.models import ProtoPTST#你训练的那个“分段/shapelet/prototype”的模型就在这里

# Add parent directory to sys.path for local imports
from shapeX import ScoreSubsequences#给定分类模型 + 解释器模型，对每个时间点/片段算 saliency 分数
try:
    from tslearn.clustering import TimeSeriesKMeans
except Exception:
    TimeSeriesKMeans = None



class Exp_Basic(object):#实验的通用骨架（设备 + 建模接口）
    """Base experiment wrapper providing device setup and model interface."""

    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "ProtoPTST":  ProtoPTST, #ProtoPTST
            
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)#创建模型（但具体怎么建让子类实现：_build_model()）

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """Acquire compute device (GPU/CPU) based on configuration."""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Classification(Exp_Basic):
    """Classification experiment for training ProtoPTST explainer."""

    def __init__(self, args, data_dict):
        self.data_dict = data_dict
        super().__init__(args)
        

        self.swa_model = optim.swa_utils.AveragedModel(self.model)
        self.swa = args.swa
       

    def _build_model(self):#建 ProtoPTST 模型，并把数据的关键信息写回 args
        """Build explainer model and set basic args from data."""
        # model input depends on data
        # train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(self.data_dict,flag="TEST")#用数据来决定模型输入输出的一些维度（例如 enc_in 输入通道数）。
        self.args.seq_len = self.args.seq_len  # redefine seq_len
        self.args.pred_len = 0
        # self.args.enc_in = train_data.feature_df.shape[1]
        # self.args.num_class = len(train_data.class_names)
        self.args.enc_in = self.args.enc_in  # redefine enc_in
        self.args.num_class = self.args.num_classes
        if self.args.data == "mitecg":
            self.args.num_class = 2
        
        
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    def _get_data(self, data_dict,flag):#把 data_dict 变成 DataLoader
        """Return dataset and dataloader for the requested split flag."""
        
        if self.args.meta_dataset == 'ucr':
            data_set = data_dict[2]
            data_loader =  DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True)
            return data_set, data_loader
        else:
            random.seed(self.args.seed)
            data_set=data_dict[flag] 
            data_loader =  DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True)
            return data_set, data_loader

    def _select_optimizer(self):#把可替换的东西封装成单独函数。
        """Create optimizer for training."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """Classification loss function."""
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _kmeans_init_shapelets(self, train_data):
        """Initialize shapelets via KMeans on random sliding windows."""
        if not getattr(self.args, "use_shapelet_layer", False):
            return None
        if TimeSeriesKMeans is None:
            print("[KMeans init] tslearn not available; keep random initialization.")
            return None
        if not hasattr(train_data, "X"):
            print("[KMeans init] train_data has no attribute X; skip.")
            return None

        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if not hasattr(model, "shapelet_layer"):
            print("[KMeans init] model has no shapelet_layer; skip.")
            return None

        X = train_data.X
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        X = X.detach().cpu()
        if X.dim() == 2:
            X = X.unsqueeze(-1)
        if X.dim() != 3:
            print(f"[KMeans init] unexpected X shape {tuple(X.shape)}; skip.")
            return None

        N, T, C = X.shape
        L = int(self.args.prototype_len)
        if T < L:
            print(f"[KMeans init] seq_len {T} < prototype_len {L}; skip.")
            return None

        max_windows = N * max(1, T - L + 1)
        n_segments = min(10000, max_windows)

        # Extract random sliding windows: (N, T-L+1, L, C) -> (N*(T-L+1), L, C)
        windows = X.unfold(1, L, 1).contiguous().view(-1, L, C)
        if windows.shape[0] > n_segments:
            idx = torch.randperm(windows.shape[0])[:n_segments]
            windows = windows[idx]

        segments = windows.numpy()
        kmeans = TimeSeriesKMeans(
            n_clusters=int(self.args.num_prototypes),
            metric="euclidean",
            max_iter=50,
        )
        kmeans.fit(segments)
        centers = kmeans.cluster_centers_  # (P, L, C)
        centers_t = torch.tensor(centers, dtype=model.shapelet_layer.shapelets.dtype)
        model.shapelet_layer.set_shapelets(centers_t)
        print("[KMeans init] shapelets initialized via TimeSeriesKMeans.")
        return centers_t

    def vali(self, vali_data, vali_loader, criterion):
        """Validate on a data split and compute metrics dictionary."""
        total_loss = []
        preds = []
        trues = []
        if self.swa:
            self.swa_model.eval()
        else:
            self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
#                 batch_x：时间序列数据（Tensor）

# label：分类标签（Tensor）

# padding_mask：padding 掩码（Tensor，用来告诉模型哪些位置是真数据，哪些是补齐的）

# 这就是 get_data.py 必须遵守的输出接口。
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                if self.swa:
                    outputs = self.swa_model(batch_x, padding_mask, None, None)
                else:
                    outputs , place_holder,_= self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach().cpu()
                loss = criterion(pred, label.cpu())
                total_loss.append(loss)

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes)

        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        probs_np = probs.cpu().numpy()
        trues_np = trues.flatten().cpu().numpy()

        metrics_dict = {
            "Accuracy": accuracy_score(trues_np, predictions),
            "Precision": precision_score(trues_np, predictions, average="macro", zero_division=0),
            "Recall": recall_score(trues_np, predictions, average="macro", zero_division=0),
            "F1": f1_score(trues_np, predictions, average="macro", zero_division=0),
        }

        unique_labels = np.unique(trues_np)
        if unique_labels.size < 2:
            print("[vali] Only one class present in y_true; set AUROC/AUPRC to 0 for this split.")
            metrics_dict["AUROC"] = 0.0
            metrics_dict["AUPRC"] = 0.0
        else:
            trues_onehot = (
                torch.nn.functional.one_hot(
                    torch.tensor(trues_np, dtype=torch.long),
                    num_classes=self.args.num_class,
                )
                .float()
                .cpu()
                .numpy()
            )
            metrics_dict["AUROC"] = roc_auc_score(trues_onehot, probs_np, multi_class="ovr")
            metrics_dict["AUPRC"] = average_precision_score(trues_onehot, probs_np, average="macro")

        if self.swa:
            self.swa_model.train()
        else:
            self.model.train()
        return total_loss, metrics_dict

    def train(self):#训练主循环（ProtoPTST + 原分类 loss + prototype loss）
        """Train explainer model with early stopping and optional SWA updates."""
        train_data, train_loader = self._get_data(self.data_dict,flag="TRAIN")
        vali_data, vali_loader = self._get_data(self.data_dict,flag="VAL")
        test_data, test_loader = self._get_data(self.data_dict,flag="TEST")
        path = self.args.saving_path
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self._kmeans_init_shapelets(train_data)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                
                outputs, place_holder_1, prototype= self.model(batch_x, padding_mask, None, None)#   分类输出（logits）\某种中间特征/激活（供 prototype_loss 用）\          prototype 本体（shapelet 字典/原型）   
                prototype_loss =  self.model.seg_prototype_loss ( place_holder_1,batch_x,prototype,outputs)

                global place_holder 
                place_holder = place_holder_1    
                loss =  criterion(outputs, label)+ 10*prototype_loss # + activation_loss
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("prototype_loss:", prototype_loss, "loss:", criterion(outputs, label))
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            self.swa_model.update_parameters(self.model)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
            )
            early_stopping(
                -val_metrics_dict["F1"],
                self.swa_model if self.swa else self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)"""

        best_model_path = path 
        if self.swa:
            self.swa_model.load_state_dict(torch.load(best_model_path),map_location=self.device)
        else:
            #self.model.load_state_dict(torch.load(best_model_path),map_location=self.device)
            state_dict = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        return self.model

    def test(self, test=0):
        """Evaluate model on validation and test splits and log metrics to file."""
        vali_data, vali_loader = self._get_data(self.data_dict,flag="VAL")
        test_data, test_loader = self._get_data(self.data_dict,flag="TEST")
        if test:
            print("loading model")
            
            path = self.saving_path

            model_path = path 
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            if self.swa:
                self.swa_model.load_state_dict(torch.load(model_path))
            else:
                self.model.load_state_dict(torch.load(model_path))

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        # result save
        folder_path = (
            "./results/"
            + self.args.task_name
            + "/"
            + self.args.model_id
            + "/"
            + self.args.model
            + "/"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        file_name = "result_classification.txt"
        f = open(os.path.join(folder_path, file_name), "a")
        f.write(f"{self.args.dataset}  \n")
        f.write(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return



class ShapeXPipline:#总调度者（对 runner/api 的接口） 这是你外部真正调用的东西。
    """High-level pipeline orchestrator for ShapeX experiments."""
    
    def __init__(self,config):
        self.args = get_args(config)
        
        # Root is provided by runner/config; allow None
        self.root_dir = getattr(self.args, 'root_path', None)

        # Defer data loading until train/test to respect flags set by runner
        self.data_dict = None

        # Explainer checkpoint path
        base_root = self.root_dir or "."
        self.saving_path = f"{base_root}/checkpoints/explainer/{self.args.dataset_name}_shapex.pt"
        self.args.saving_path = self.saving_path
        
        
    def get_class_model(self, X):
        """Instantiate or load the classification backbone used for scoring."""
        
        if self.args.class_model_type=="cnn":
            class_model = CNN(
            d_inp = X[0].shape[-1],
            n_classes = self.args.num_classes,)
            
        elif self.args.class_model_type=="lstm":
                class_model = LSTM(
            d_inp = X[0].shape[-1],
            n_classes = self.args.num_classes,
        )
                
        else:
            class_model=TransformerMVTS(
        
                    d_inp = X.shape[-1],
                    max_len = X.shape[1],
                    nlayers = 1,
                    n_classes = self.args.num_classes,
                    trans_dim_feedforward = 64,
                    trans_dropout = 0.1,
                    d_pe = 16,
                    stronger_clf_head = False,
                    norm_embedding = True,
                        )
        

        if self.args.data == "mitecg":
            # Use fixed transformer classifier checkpoint for mitecg
            class_model = TransformerMVTS(
                    d_inp = X.shape[-1],
                    max_len = X.shape[1],
                    n_classes = self.args.num_classes,
                    trans_dim_feedforward = 64,
                    trans_dropout = 0.1,
                    d_pe = 16,
                    stronger_clf_head = False,
                    norm_embedding = True,
            )
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models', 'mitecg_transformer.pt')
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            class_model = class_model.to(self.args.device)
            class_model.eval()

        # New datasets (mcce/mcch/mtce/mtch) use local classification models
        if self.args.data in ["mcce","mcch","mtce","mtch"]:
            class_model= TransformerMVTS(
                    d_inp = X.shape[-1],
                    max_len = X.shape[1],
                    n_classes = 4,
                    trans_dim_feedforward = 16,
                    trans_dropout = 0.1,
                    d_pe = 16,)
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models', f"{self.args.data}_transformer.pt")
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            class_model=class_model.to(self.args.device)
            class_model.eval()

        elif self.args.meta_dataset == "ucr":
            # UCR classifiers stored under classification_models
            model_path = os.path.join(self.args.root_path or '.', 'checkpoints', 'classification_models', f"{self.args.dataset}.pt")
            class_model.load_state_dict(torch.load(model_path, map_location=self.args.device))
            
        return class_model
                                             
        
    def train_shapex(self):
        """Train the explainer model on the configured dataset."""
        
        args = self.args
        print(args)
        trained_model=None
                
        if args.is_training:
            # Use timestamped checkpoint name for training runs
            dm = str(getattr(self.args, "dist_measure", "na"))
            zn = int(bool(getattr(self.args, "shapelet_znorm", False)))
            tmp = getattr(self.args, "shapelet_temperature", "na")
            tmp_str = str(tmp).replace(".", "p")
            base_root = self.root_dir or "."
            self.saving_path = (
                f"{base_root}/checkpoints/explainer/"
                f"{self.args.dataset_name}_shapex_{dm}_zn{zn}_t{tmp_str}.pt"
            )
            self.args.saving_path = self.saving_path
            print(f"[Checkpoint] Saving to: {self.saving_path}")
            # Load data now with current flags
            self.data_dict = get_saliency_data(self.args, return_dict=True)#和 get_data.py 的接口,要求 get_saliency_data(args, return_dict=True) 返回 data_dict（含 TRAIN/VAL/TEST）
            for ii in range(args.itr):
                seed = 41 + ii
                random.seed(seed)
                os.environ["PYTHONHASHSEED"] = str(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                # setting record of experiments
                args.seed = seed
                
                ############ Train segmentation model
                exp = Exp_Classification(args, self.data_dict)
                
                exp.train()
                
                model=exp.model
                
                
    def load_seg_model(self):#从磁盘加载解释器（ProtoPTST）
        """Load a saved segmentation/explainer checkpoint from disk."""
        
        #seg_model = ProtoPTST.Model(args).float()
        seg_model = ProtoPTST.Model(self.args).float()

        state = torch.load(self.saving_path, map_location=self.args.device)
        # Backward compatibility: old ckpt uses "prototype_vectors" while new model uses shapelet_layer
        if getattr(self.args, "use_shapelet_layer", False):
            if "shapelet_layer.shapelets" not in state and "prototype_vectors" in state:
                state["shapelet_layer.shapelets"] = state.pop("prototype_vectors")
            # Fix possible layout mismatch: [P, L, C] vs [P, C, L]
            if "shapelet_layer.shapelets" in state:
                ckpt_w = state["shapelet_layer.shapelets"]
                model_w = seg_model.state_dict()["shapelet_layer.shapelets"]
                if ckpt_w.shape != model_w.shape:
                    # checkpoint: [P, L, C] -> model expects [P, C, L]
                    if ckpt_w.dim() == 3 and ckpt_w.permute(0, 2, 1).shape == model_w.shape:
                        state["shapelet_layer.shapelets"] = ckpt_w.permute(0, 2, 1).contiguous()
        else:
            if "prototype_vectors" not in state and "shapelet_layer.shapelets" in state:
                state["prototype_vectors"] = state.pop("shapelet_layer.shapelets")

        seg_model.load_state_dict(state, strict=False)#然后 load_state_dict 读 checkpoint
        seg_model=seg_model.to(self.args.device)
            
        return seg_model
     
        
    def eval_shapex(self):
        """Evaluate saliency explanations against ground-truth annotations."""
        # only use when there are ground truth explanations
            
        seg_model=self.load_seg_model()

        ############ Load test data and classification model

    
        self.args.saliency = True
        
        X ,y,times, gt_exps = get_saliency_data(self.args,return_dict=False)#读测试数据 + ground truth explanations
        X, y, times, gt_exps = X.cpu(), y.cpu(), times.cpu(), gt_exps.cpu()
        class_model = self.get_class_model(X)#加载分类器 class_model  它最终返回一个可 forward(X) 的分类模型（用于解释打分）
        
        ############ compute saliency scores
        Scorer = ScoreSubsequences(self.args,class_model, seg_model)#用 ScoreSubsequences 生成 saliency score
        score = Scorer.get_score_as_GTEXP(X)#用 ground_truth_xai_eval 对比 gt_exps，输出指标
    
   
        # shapes are (N, T, 1) and (N, T, 1)
        
        
        results_dict = ground_truth_xai_eval(score, gt_exps)#输出一个 dict，每个 key 是指标名，value 是一个列表/数组（每个样本一个值），所以你会看到最后算 mean/std。
        for k, v in results_dict.items():
            print('\t{} = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))
            
        
        with open("results.txt", "a") as f:  # "a" means append mode
            

            f.write(f"{self.args.data}_slen{self.args.seq_len}_npro{self.args.num_prototypes}_prototype_len{self.args.prototype_len}")
            for k, v in results_dict.items():
                f.write(f' {k}:{np.mean(v):.4f}-{(np.std(v) / np.sqrt(len(v))):.4f}')
            f.write("\n") 
        
        
    def get_saliency_score(self,X=None):
        """Compute saliency scores for provided data or test split (N, T, 1)."""
        
        seg_model=self.load_seg_model()
        

        ############ Load test data and classification model

        self.args.saliency = False
        
        train, val, test, _= get_saliency_data(self.args,return_dict=False)
        
        if X is None:
            X, y, times = test[0], test[2], test[1]
        
        X=X.to("cpu")
        self.args.device="cpu"
   
        
        class_model = self.get_class_model(X)
        class_model.to("cpu")
        
        ############ compute saliency scores
        Scorer = ScoreSubsequences(self.args,class_model, seg_model)
        score = Scorer.get_score_as_GTEXP(X) # (N,T,1)
  
        return score
    
    
    
    
    
    
    
    


    
    
    
def get_args(config):
    """Construct argument namespace from the provided config.

    Dataset-specific YAML (if present) overrides sensible defaults for
    num_classes, seq_len, prototype length, and number of prototypes.
    """
    # dataset-sensitive defaults: load from dataset-specific YAML if present
    dataset_name = config.dataset.name

# 使用你传入的 root_path 自动拼接
    root = config.base.root_dir
    ds_cfg_path = os.path.join(root, "configs", f"{dataset_name}.yaml")
    
    # 打印一下路径，确认它找对了
    print(f"DEBUG: Trying to load config from: {ds_cfg_path}")
    # sensible defaults; will be overridden by dataset YAML if present
    NUM_CLASSES = 2
    SEQ_LEN = 500
    PROTO_LEN = 30
    N_PROTOS = 1
    if os.path.exists(ds_cfg_path):
        
        ds_cfg = OmegaConf.load(ds_cfg_path)
        if dataset_name in ds_cfg:
            block = ds_cfg[dataset_name]
            NUM_CLASSES = int(block.get('num_classes', 2))
            SEQ_LEN = int(block.get('seq_len', 500))
            PROTO_LEN = int(block.get('proto_len', 30))
            N_PROTOS = int(block.get('num_prototypes', 1))
        
    
    
    parser = argparse.ArgumentParser()
    # Experiment settings   
    # Task settings
    parser.add_argument("--task_name", type=str, default="classification")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="dataset_name")  # Replace 'dataset_name' with the actual variable value
    parser.add_argument("--model", type=str, default="ProtoPTST")
    

    # Data settings
    parser.add_argument("--data", type=str, default=config.dataset.name)  # Replace 'dataset_name' with the actual variable value
    parser.add_argument("--dataset_name", type=str, default=config.dataset.name)
    parser.add_argument("--root_path", type=str, default=config.base.root_dir)
    parser.add_argument("--two_class", type=int, default=0)
    parser.add_argument("--saliency", type=bool, default=False)
    parser.add_argument("--s_no", type=int, default=0)
    parser.add_argument("--num_classes", type=int, default=NUM_CLASSES)


    # Model settings for classification
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--e_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", type=bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", type=bool, default=True)
    parser.add_argument("--no_inter_attn", type=bool, default=False)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--patch_len", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--patch_len_list", type=list, default=[2, 4, 8])
    parser.add_argument("--single_channel", type=bool, default=False)
    parser.add_argument("--augmentations", type=str, default="flip,shuffle,jitter,mask,drop")

    # Optimization settings
    parser.add_argument("--batch_size", type=int, default=16)  # Replace 'BATCH_SIZE' with the actual variable value
    parser.add_argument("--train_epochs", type=int, default=10) # 100
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--patience", type=int, default=3) # 5
    parser.add_argument("--des", type=str, default="Exp")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", type=bool, default=False)
    parser.add_argument("--swa", type=bool, default=False)
    parser.add_argument("--d_layers", type=int, default=1)

    # GPU settings
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--gpu", type=str, default=0)  # Replace 'gpu_n' with the actual variable value
    parser.add_argument("--use_multi_gpu", type=bool, default=False)
    parser.add_argument("--devices", type=str, default=1)  # Replace 'gpu_n' with the actual variable value
    parser.add_argument("--device", type=str, default="cuda:0")  # Replace 'gpu_n' with the actual variable value

    # De-stationary projector parameters
    parser.add_argument("--p_hidden_dims", type=list, default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--num_workers", type=int, default=1)

    # Building model args
    parser.add_argument("--pred_len", type=int, default=0)
    parser.add_argument("--enc_in", type=int, default=1)  # Replace 'CHANNEL' with the actual variable value
    parser.add_argument("--num_class", type=int, default=NUM_CLASSES)  # Replace 'CLASS_NUMBER' with the actual variable value

    # ProtoPTST args
    parser.add_argument("--num_prototypes", "-n_pro", type=int, default=N_PROTOS)  # YAML-overridable
    parser.add_argument("--prototype_len","-pro_len", type=int, default=PROTO_LEN)  # Replace 'proto_len' with the actual variable value
    parser.add_argument("--prototype_init",default= "kaiming",type=str)
    parser.add_argument("--prototype_activation",default= "linear",type=str)
    parser.add_argument("--ablation",default= "none",type=str,help=["no_matching_loss","no_variances_loss",'no_prototype_layer'])
    parser.add_argument("--class_model_type",default="cnn")
    parser.add_argument("--equal_seg_len",default=0, type= int)
    parser.add_argument("--seq_method",default=None, type= str)
    
    # 新增 LearningShapeletsSeg 相关参数
    parser.add_argument("--use_shapelet_layer", type=bool, default=True, 
                       help="Use LearningShapeletsSeg instead of conv1d")
    parser.add_argument("--dist_measure", type=str, default='cosine',
                       choices=['euclidean', 'cosine', 'correlation'],
                       help="Distance measure for shapelet matching")
    parser.add_argument("--shapelet_znorm", type=bool, default=True,
                       help="Apply z-normalization to shapelets")
    parser.add_argument("--shapelet_temperature", type=float, default=1.0,
                       help="Temperature for softmax in shapelet layer")
    
    ############ Parse arguments
    # do not read CLI; stick to defaults and YAML overrides
    args = parser.parse_args(args=[])
    ## print(args)
    print("#"*20)
    print(args)
    print("#"*20)
    ############ Load data
    
    # Add dataset to args
    args.dataset = config.dataset.name
    args.meta_dataset = config.dataset.meta_dataset
    
    # Inject base root path only (no other path params)
    base_cfg = getattr(config, 'base', {})
    args.root_path = getattr(base_cfg, 'root_dir', None)
    
    return args

## Module is intended to be imported by orchestration scripts; no CLI entry.
