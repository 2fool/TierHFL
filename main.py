import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import sys
import random
import argparse
import logging
import wandb
import copy
import warnings
from collections import defaultdict
import torchvision
import torchvision.transforms as transforms
import math

# 忽略警告
warnings.filterwarnings("ignore")

# 确保项目根目录在Python路径中 - Kaggle兼容
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"项目根目录: {project_root}")
print(f"当前工作目录: {os.getcwd()}")

# 扁平导入自定义模块
from model.resnet import EnhancedServerModel, TierAwareClientModel, ImprovedGlobalClassifier
from utils.tierhfl_aggregator import LayeredAggregator
from utils.tierhfl_client import TierHFLClientManager
from utils.tierhfl_loss import EnhancedStagedLoss
from analyze.tierhfl_analyze import validate_server_effectiveness
from analyze.diagnostic_monitor import EnhancedTierHFLDiagnosticMonitor

# === logging setup (放在 main.py 的 import 之后) ===
import logging, sys
from pathlib import Path
from datetime import datetime

def setup_logging(run_name: str = "run",
                  log_dir: str = "logs",
                  console_level=logging.INFO,
                  file_level=logging.DEBUG) -> str:
    """
    同时输出到控制台 + 文件；强制覆盖已有 handler，防止 basicConfig 失效。
    返回日志文件路径字符串。
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{run_name}_{ts}.log"

    fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d - %(message)s"
    datefmt = "%H:%M:%S"

    logging.basicConfig(
        level=file_level,                       # 根 logger 记 DEBUG 及以上到文件
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),  # 控制台
            logging.FileHandler(log_path, mode="w", encoding="utf-8")  # 文件
        ],
        force=True                              # 关键：覆盖之前任何 handler/basicConfig
    )

    # 控制台只打 INFO+；文件仍是 DEBUG+
    logging.getLogger().handlers[0].setLevel(console_level)

    # 确保命名 logger（如 "TierHFL" 等）冒泡到根 logger，从而也写文件
    for name in ["TierHFL", "analyze", "utils", "__main__"]:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True

    logging.info(f"日志已写入: {log_path}")
    return str(log_path)


# ========= EnhancedSerialTrainer (GPU-ready) =========
class EnhancedSerialTrainer:
    def __init__(self, client_manager, server_model, global_classifier, device="auto", use_amp=False):
        self.client_manager = client_manager
        self.server_model = server_model
        self.global_classifier = global_classifier

        # 设备选择
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 把服务器侧两个模型搬到设备上
        self.server_model.to(self.device)
        self.global_classifier.to(self.device)

        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.client_models = {}
        self.cluster_map = {}
        self.cluster_server_models = {}
        self.cluster_global_classifiers = {}

        from utils.tierhfl_aggregator import LayeredAggregator
        # 让聚合计算也在相同 device 上做（减少 CPU<->GPU 拖拽）
        self.layered_aggregator = LayeredAggregator(device=str(self.device))

        from utils.tierhfl_loss import EnhancedStagedLoss
        self.enhanced_loss = EnhancedStagedLoss(
            ls_eps=0.05,       # 降低标签平滑，减少小类信息被抹平
            entropy_coeff=2e-3  # 增强全局头的输出熵约束，防止单类偏好
        )
        
        # 客户端性能追踪（用于分层采样）
        self.client_performance = defaultdict(lambda: {'speed_score': 1.0, 'accuracy_ema': 0.0})
        
        # FedProx参数
        self.mu = 0.0  # 将在main函数中设置

    def _prepare_models_for_round(self, server_model, global_classifier):
        """本轮只调用一次，把聚合后的权重复制到trainer持有的模型上"""
        self.server_model.load_state_dict(server_model.state_dict())
        self.global_classifier.load_state_dict(global_classifier.state_dict())
        # ⚠️ 注意：在初始阶段需要train()模式，不能强制eval()
        self.server_model.to(self.device)         # 由具体训练阶段决定train/eval模式
        self.global_classifier.to(self.device).train()

    def _unwrap_loader(self, loader):
        """
        如果误把 DataLoader 再次用 DataLoader 包了一层，则解包成内部的那一层；
        否则原样返回。
        """
        try:
            if isinstance(loader, DataLoader) and isinstance(getattr(loader, "dataset", None), DataLoader):
                return loader.dataset
        except Exception:
            pass
        return loader

    def register_client_models(self, client_models_dict):
        self.client_models.update(client_models_dict)

    def setup_training(self, cluster_map):
        self.cluster_map = cluster_map
        self.cluster_server_models = {}
        self.cluster_global_classifiers = {}
        # 用当前 server/global classifier 的权重作为各簇初值
        for cluster_id in cluster_map.keys():
            self.cluster_server_models[cluster_id] = copy.deepcopy(self.server_model.state_dict())
            self.cluster_global_classifiers[cluster_id] = copy.deepcopy(self.global_classifier.state_dict())
    
    def execute_round(self, round_idx, total_rounds, args, diagnostic_monitor=None):
        """执行一轮训练 - 集成增强版监控和综合分析"""
        start_time = time.time()
        
        # 监控学习率
        if diagnostic_monitor is not None:
            client_lrs = {}
            for client_id in range(len(self.client_models)):
                client = self.client_manager.get_client(client_id)
                if client:
                    client_lrs[client_id] = client.lr
            
            lr_analysis = diagnostic_monitor.monitor_learning_rates(client_lrs, round_idx)

        # 🔥 分层采样选择本轮参与的客户端
        selected_cluster_map = self.select_clients_for_round(
            self.cluster_map, args.client_fraction, round_idx
        )
        
        # 统计选择的客户端数量
        total_selected = sum(len(clients) for clients in selected_cluster_map.values())
        total_available = sum(len(clients) for clients in self.cluster_map.values())
        logging.info(f"分层采样：选择了 {total_selected}/{total_available} 个客户端参与训练")

        # 结果容器
        train_results = {}
        eval_results = {}
        shared_states = {}
        
        # 确定当前训练阶段 - 使用参数化边界而非硬编码
        if round_idx < args.initial_feature_rounds:  # 🔥 使用initial_feature_rounds参数
            training_phase = "initial"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 初始特征学习阶段")
        elif round_idx < args.initial_phase_rounds:
            training_phase = "initial"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 初始阶段")
        elif round_idx < args.initial_phase_rounds + args.alternating_phase_rounds:
            training_phase = "alternating"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 交替训练阶段")
        else:
            training_phase = "fine_tuning"
            logging.info(f"轮次 {round_idx+1}/{total_rounds} - 精细调整阶段")
        
        # 依次处理每个聚类（使用选择后的客户端）
        for cluster_id, client_ids in selected_cluster_map.items():
            logging.info(f"处理聚类 {cluster_id}, 包含 {len(client_ids)} 个客户端")
            
            # 创建聚类特定的模型
            cluster_server = copy.deepcopy(self.server_model).to(self.device)
            cluster_server.load_state_dict(self.cluster_server_models[cluster_id])
            
            cluster_classifier = copy.deepcopy(self.global_classifier).to(self.device)
            cluster_classifier.load_state_dict(self.cluster_global_classifiers[cluster_id])
            
            # === 监控聚类模型稳定性 ===
            if diagnostic_monitor is not None:
                diagnostic_monitor.monitor_model_stability_fixed(
                    cluster_server.state_dict(), round_idx, f"server_cluster_{cluster_id}"
                )
                diagnostic_monitor.monitor_model_stability_fixed(
                    cluster_classifier.state_dict(), round_idx, f"classifier_cluster_{cluster_id}"
                )
            
            # 处理聚类中的每个客户端
            for client_id in client_ids:
                client = self.client_manager.get_client(client_id)
                if not client or client_id not in self.client_models:
                    continue
                
                logging.info(f"训练客户端 {client_id} (Tier: {client.tier})")
                
                client_model = self.client_models[client_id].to(self.device)
                client.model = client_model
                
                # === 监控客户端模型稳定性 ===
                if diagnostic_monitor is not None:
                    diagnostic_monitor.monitor_model_stability_fixed(
                        client_model.state_dict(), round_idx, f"client_{client_id}"
                    )
                
                # 根据训练阶段执行训练
                if training_phase == "initial":
                    train_result = self._train_initial_phase_enhanced(
                        client_id,
                        client,
                        cluster_server,      # 或者你的变量名 server_model
                        cluster_classifier, # 或者你的变量名 global_classifier
                        round_idx,
                        diagnostic_monitor
                    )

                    
                elif training_phase == "alternating":
                    train_result = self._train_alternating_phase_enhanced(
                        client, client_model, cluster_server, cluster_classifier,
                        round_idx, total_rounds, diagnostic_monitor, args)
                    
                else:  # fine_tuning
                    train_result = self._train_fine_tuning_phase_enhanced(
                        client, client_model, cluster_server, cluster_classifier,
                        round_idx, total_rounds, diagnostic_monitor, args)
                
                # 保存结果
                train_results[client_id] = train_result
                
                # 🔥 更新客户端性能指标
                training_time = train_result.get('time_cost', 1.0)
                accuracy = train_result.get('global_accuracy', train_result.get('local_accuracy', 0.0))
                self.update_client_performance(client_id, training_time, accuracy)
                
                # 评估客户端
                eval_result = self._evaluate_client(
                    client, client_model, cluster_server, cluster_classifier)
                eval_result['cluster_id'] = cluster_id  # 添加聚类ID
                eval_results[client_id] = eval_result
                
                # 保存共享层状态
                shared_state = {}
                for name, param in client_model.named_parameters():
                    if 'shared_base' in name:
                        shared_state[name] = param.data.clone().cpu()
                shared_states[client_id] = shared_state
                
                # 更新客户端模型
                self.client_models[client_id] = client_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            
            # 保存聚类模型
            self.cluster_server_models[cluster_id] = cluster_server.cpu().state_dict()
            self.cluster_global_classifiers[cluster_id] = cluster_classifier.cpu().state_dict()
        
        # 计算总训练时间
        training_time = time.time() - start_time
        
        # === 生成综合诊断报告 ===
        if diagnostic_monitor is not None:
            comprehensive_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            
            # 记录关键发现
            if comprehensive_report['overall_health'] == 'critical':
                logging.error(f"轮次{round_idx+1}: 检测到严重问题!")
                for issue in comprehensive_report['critical_issues']:
                    logging.error(f"  - {issue}")
            
            if comprehensive_report['recommendations']:
                logging.info(f"轮次{round_idx+1} 优化建议:")
                for rec in comprehensive_report['recommendations']:
                    logging.info(f"  - {rec}")
        
        return train_results, eval_results, shared_states, training_time

    # === 阶段1：仅训练“全局路径”（服务器特征 + 全局分类器）===
    # 在 class EnhancedSerialTrainer 内部，完整替换这个函数
    def _train_initial_phase_enhanced(
        self, client_id, client, server_model, global_classifier, round_idx, diagnostic_monitor=None
    ):
        # ---- 统一设备 ----
        client_model = self.client_models[client_id]
        client_model.to(self.device).train()
        self.server_model.load_state_dict(server_model.state_dict())
        self.global_classifier.load_state_dict(global_classifier.state_dict())
        
        # 🔥 关键修复：让server_model也参与训练，而不是eval()
        self.server_model.to(self.device).train()        # 从eval()改为train()
        self.global_classifier.to(self.device).train()

        # 🔥 关键修复：为server_model和global_classifier分别创建优化器
        opt_server = torch.optim.SGD(
            self.server_model.parameters(), lr=client.lr, momentum=0.9, weight_decay=5e-4
        )
        opt_classifier = torch.optim.SGD(
            self.global_classifier.parameters(), lr=client.lr, momentum=0.9, weight_decay=5e-4
        )

        running_loss, total, correct = 0.0, 0, 0
        use_amp = (self.device.type == "cuda")

        for data, target in client.train_data:
            data   = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            # 🔥 关键修复：两个优化器都要清零梯度
            opt_server.zero_grad(set_to_none=True)
            opt_classifier.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                local_logits, shared_features, _ = client_model(data)
                server_features = self.server_model(shared_features)
                global_logits   = self.global_classifier(server_features)
                total_loss, _, _ = self.enhanced_loss.stage1_loss(
                    global_logits, target, shared_features=shared_features
                )

            # 🔥 关键修复：AMP支持两个优化器
            if use_amp:
                self.scaler.scale(total_loss).backward()
                # 先unscale再做梯度裁剪
                self.scaler.unscale_(opt_server)
                self.scaler.unscale_(opt_classifier)
                torch.nn.utils.clip_grad_norm_(list(self.server_model.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(list(self.global_classifier.parameters()), 1.0)
                # 两个优化器都要step
                self.scaler.step(opt_server)
                self.scaler.step(opt_classifier)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.server_model.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(list(self.global_classifier.parameters()), 1.0)
                opt_server.step()
                opt_classifier.step()

            running_loss += float(total_loss.detach())
            with torch.no_grad():
                pred = global_logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total   += target.size(0)

        train_acc = 100.0 * correct / max(1, total)
        train_samples = len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else total
        return {"train_loss": running_loss / max(1, len(client.train_data)), "train_acc": train_acc, "train_samples": train_samples}

    def _train_alternating_phase_enhanced(
        self, client, client_model, server_model, classifier, round_idx, total_rounds, diagnostic_monitor=None, args=None
    ):
        import time
        from torch.nn.utils import clip_grad_norm_
        from torch.optim.lr_scheduler import CosineAnnealingLR

        start = time.time()
        # ---- 统一设备 ----
        client_model.to(self.device).train()
        server_model.to(self.device).train()
        classifier.to(self.device).train()
        use_amp = (self.device.type == "cuda")

        # 允许全部分支训练（共享+个性化+本地头+服务器+全局头）
        for p in client_model.parameters():
            p.requires_grad = True

        # 参数分组
        shared_params   = [p for n, p in client_model.named_parameters() if ('shared_base' in n and p.requires_grad)]
        personal_params = [p for n, p in client_model.named_parameters() if ('shared_base' not in n and p.requires_grad)]
        server_params   = list(server_model.parameters())
        global_params   = list(classifier.parameters())

        # 优化器（在 to(device) 之后）
        opt_shared    = torch.optim.SGD(shared_params,   lr=client.lr, momentum=0.9, weight_decay=client.wd) if shared_params else None
        opt_personal  = torch.optim.SGD(personal_params, lr=client.lr, momentum=0.9, weight_decay=client.wd) if personal_params else None
        opt_server    = torch.optim.SGD(server_params,   lr=client.lr, momentum=0.9, weight_decay=client.wd)
        opt_global    = torch.optim.SGD(global_params,   lr=client.lr, momentum=0.9, weight_decay=client.wd)

        sch_shared    = CosineAnnealingLR(opt_shared,   T_max=max(1, client.local_epochs), eta_min=0.0) if opt_shared else None
        sch_personal  = CosineAnnealingLR(opt_personal, T_max=max(1, client.local_epochs), eta_min=0.0) if opt_personal else None
        sch_server    = CosineAnnealingLR(opt_server,   T_max=max(1, client.local_epochs), eta_min=0.0)
        sch_global    = CosineAnnealingLR(opt_global,   T_max=max(1, client.local_epochs), eta_min=0.0)

        # 🔥 使用参数控制的动态alpha调整
        if args is not None:
            progress = round_idx / max(1, total_rounds)
            alpha = args.init_alpha - (args.init_alpha - args.min_alpha) * progress
        else:
            # 回退到默认值
            progress = round_idx / max(1, total_rounds)
            alpha = 0.6 - (0.6 - 0.4) * progress

        stat = {'total_loss': 0.0, 'batch_count': 0, 'local_correct': 0, 'global_correct': 0, 'total': 0}

        for _ in range(client.local_epochs):
            for data, target in client.train_data:
                data   = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                if opt_shared:   opt_shared.zero_grad(set_to_none=True)
                if opt_personal: opt_personal.zero_grad(set_to_none=True)
                opt_server.zero_grad(set_to_none=True)
                opt_global.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    local_logits, shared_features, personal_features = client_model(data)
                    server_features = server_model(shared_features)
                    global_logits   = classifier(server_features)

                    total_loss, local_loss, global_loss, balance_loss = self.enhanced_loss.stage2_3_loss(
                        local_logits, global_logits, target,
                        personal_gradients=None, global_gradients=None,
                        shared_features=shared_features, alpha=alpha
                    )

                if use_amp:
                    self.scaler.scale(total_loss).backward()
                    # 可选：共享层梯度投影
                    # self.enhanced_loss.apply_gradient_projection(client_model, local_loss, global_loss, alpha_stage=alpha)
                    if opt_shared:   self.scaler.unscale_(opt_shared);   clip_grad_norm_(shared_params,   1.0)
                    if opt_personal: self.scaler.unscale_(opt_personal); clip_grad_norm_(personal_params, 1.0)
                    self.scaler.unscale_(opt_server); clip_grad_norm_(server_params, 1.0)
                    self.scaler.unscale_(opt_global); clip_grad_norm_(global_params, 1.0)

                    if opt_shared:   self.scaler.step(opt_shared)
                    if opt_personal: self.scaler.step(opt_personal)
                    self.scaler.step(opt_server)
                    self.scaler.step(opt_global)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    if opt_shared:   clip_grad_norm_(shared_params,   1.0)
                    if opt_personal: clip_grad_norm_(personal_params, 1.0)
                    clip_grad_norm_(server_params, 1.0)
                    clip_grad_norm_(global_params, 1.0)

                    if opt_shared:   opt_shared.step()
                    if opt_personal: opt_personal.step()
                    opt_server.step()
                    opt_global.step()

                stat['total_loss']   += float(total_loss.item()); stat['batch_count'] += 1
                with torch.no_grad():
                    stat['local_correct']  += (local_logits.argmax(1)  == target).sum().item()
                    stat['global_correct'] += (global_logits.argmax(1) == target).sum().item()
                    stat['total']          += target.size(0)

            if sch_shared:   sch_shared.step()
            if sch_personal: sch_personal.step()
            sch_server.step(); sch_global.step()

        avg_loss   = stat['total_loss'] / max(1, stat['batch_count'])
        acc_local  = 100.0 * stat['local_correct']  / max(1, stat['total'])
        acc_global = 100.0 * stat['global_correct'] / max(1, stat['total'])
        train_samples = len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else stat['total']

        return {
            'train_loss': avg_loss,
            'local_accuracy': acc_local,
            'global_accuracy': acc_global,
            'time_cost': time.time() - start,
            'train_samples': train_samples
        }


    
    
    def _train_fine_tuning_phase_enhanced(self, client, client_model, server_model, global_classifier,
                                          round_idx, total_rounds, diagnostic_monitor=None, args=None):
        import time
        from torch.nn.utils import clip_grad_norm_
        from torch.optim.lr_scheduler import CosineAnnealingLR
        start = time.time()

        # 使用trainer的模型而非传入参数，确保权重累积
        client_model.to(self.device)
        self.server_model.to(self.device)
        self.global_classifier.to(self.device)

        # 冻结共享层，主训个性化与本地头；server/global 小步微调
        for n,p in client_model.named_parameters():
            p.requires_grad = ('shared_base' not in n)

        client_model.train(); self.server_model.train(); self.global_classifier.train()

        personal_params = [p for p in client_model.parameters() if p.requires_grad]
        opt_personal = torch.optim.SGD(personal_params, lr=client.lr, momentum=0.9, weight_decay=client.wd)
        opt_server   = torch.optim.SGD(self.server_model.parameters(),  lr=client.lr*0.2, momentum=0.9, weight_decay=client.wd)
        opt_global   = torch.optim.SGD(self.global_classifier.parameters(), lr=client.lr*0.2, momentum=0.9, weight_decay=client.wd)

        sch_personal = CosineAnnealingLR(opt_personal, T_max=max(1, client.local_epochs), eta_min=0.0)
        sch_server   = CosineAnnealingLR(opt_server,   T_max=max(1, client.local_epochs), eta_min=0.0)
        sch_global   = CosineAnnealingLR(opt_global,   T_max=max(1, client.local_epochs), eta_min=0.0)

        stat = {'total_loss':0.0,'batch_count':0,'local_correct':0,'global_correct':0,'total':0}
        # 🔥 使用参数控制的动态alpha调整
        if args is not None:
            progress = round_idx / max(1, total_rounds)
            alpha = args.init_alpha - (args.init_alpha - args.min_alpha) * progress
        else:
            # 回退到默认值
            progress = round_idx / max(1, total_rounds)
            alpha = 0.6 - (0.6 - 0.4) * progress

        train_loader = self._unwrap_loader(client.train_data)

        for _ in range(client.local_epochs):
            for data, target in train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                opt_personal.zero_grad(); opt_server.zero_grad(); opt_global.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.use_amp and torch.cuda.is_available()):
                    local_logits, shared_features, personal_features = client_model(data)
                    server_features = self.server_model(shared_features)
                    global_logits   = self.global_classifier(server_features)
                    total_loss, local_loss, global_loss, balance_loss = self.enhanced_loss.stage2_3_loss(
                        local_logits, global_logits, target,
                        personal_gradients=None, global_gradients=None,
                        shared_features=shared_features, alpha=alpha
                    )

                if self.use_amp and torch.cuda.is_available():
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(opt_personal); self.scaler.unscale_(opt_server); self.scaler.unscale_(opt_global)
                else:
                    total_loss.backward()

                clip_grad_norm_(personal_params, max_norm=1.0)
                for group in [self.server_model.parameters(), self.global_classifier.parameters()]:
                    clip_grad_norm_(list(group), max_norm=1.0)

                if self.use_amp and torch.cuda.is_available():
                    self.scaler.step(opt_personal); self.scaler.step(opt_server); self.scaler.step(opt_global); self.scaler.update()
                else:
                    opt_personal.step(); opt_server.step(); opt_global.step()

                stat['total_loss'] += float(total_loss.item()); stat['batch_count'] += 1
                with torch.no_grad():
                    stat['local_correct']  += (local_logits.argmax(1)  == target).sum().item()
                    stat['global_correct'] += (global_logits.argmax(1) == target).sum().item()
                    stat['total']          += target.size(0)

            sch_personal.step(); sch_server.step(); sch_global.step()

        avg_loss   = stat['total_loss'] / max(1, stat['batch_count'])
        acc_local  = 100.0 * stat['local_correct']  / max(1, stat['total'])
        acc_global = 100.0 * stat['global_correct'] / max(1, stat['total'])
        train_samples = len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else stat['total']
        return {'train_loss': avg_loss, 'local_accuracy': acc_local, 'global_accuracy': acc_global,
                'time_cost': time.time()-start, 'train_samples': train_samples}





    def _train_personal_path_enhanced(self, client, client_model, round_idx, total_rounds, diagnostic_monitor=None):
        import time
        import torch.nn.functional as F
        from torch.nn.utils import clip_grad_norm_
        from torch.optim.lr_scheduler import CosineAnnealingLR

        start = time.time()
        client_model.train()

        # 1) 冻结共享层，只训个性化路径 + 本地头
        for n, p in client_model.named_parameters():
            if 'shared_base' in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        trainable_params = [p for p in client_model.parameters() if p.requires_grad]
        opt_personal = torch.optim.Adam(trainable_params, lr=client.lr, weight_decay=client.wd)
        sch_personal = CosineAnnealingLR(opt_personal, T_max=max(1, client.local_epochs), eta_min=0.0)

        stat = {'local_loss': 0.0, 'correct': 0, 'total': 0, 'batch_count': 0}

        for _ in range(client.local_epochs):
            for data, target in client.train_data:
                data, target = data.to(self.device), target.to(self.device)
                opt_personal.zero_grad()
                local_logits, shared_f, personal_f = client_model(data)
                loss = F.cross_entropy(local_logits, target)
                loss.backward()
                clip_grad_norm_(trainable_params, max_norm=1.0)
                opt_personal.step()

                stat['local_loss'] += float(loss.item()); stat['batch_count'] += 1
                with torch.no_grad():
                    pred = local_logits.argmax(dim=1)
                    stat['correct'] += (pred == target).sum().item()
                    stat['total'] += target.size(0)

            sch_personal.step()

        # 解冻回去，供后续阶段使用
        for p in client_model.parameters(): p.requires_grad = True

        avg_local = stat['local_loss'] / max(1, stat['batch_count'])
        acc_local = 100.0 * stat['correct'] / max(1, stat['total'])
        train_samples = len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else stat['total']
        return {'local_loss': avg_local, 'local_accuracy': acc_local, 'time_cost': time.time() - start, 'train_samples': train_samples}


    def _train_global_path_enhanced(self, client, client_model, server_model, classifier, 
                                  shared_lr, round_idx, total_rounds, diagnostic_monitor=None, epochs=1):
        """增强版全局路径训练"""
        start_time = time.time()
        
        # 设置训练模式
        client_model.train()
        server_model.train()
        classifier.train()
        
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
            lr=shared_lr
        )
        server_optimizer = torch.optim.Adam(server_model.parameters(), lr=0.001)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        
        # 统计信息
        stats = {
            'global_loss': 0.0,
            'local_loss': 0.0,
            'balance_loss': 0.0,
            'total_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 计算alpha值
        progress = round_idx / total_rounds
        alpha = 0.3 + 0.4 * progress  # 个性化权重随训练进度增加
        
        # 训练循环
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(client.train_data):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                local_logits, shared_features, personal_features = client_model(data)
                server_features = server_model(shared_features)
                global_logits = classifier(server_features)
                
                # 计算损失
                local_loss = F.cross_entropy(local_logits, target)
                global_loss = F.cross_entropy(global_logits, target)
                
                # 计算梯度用于特征平衡损失
                # 为避免计算图复杂化，我们简化特征平衡损失
                balance_loss = torch.tensor(0.0, device=global_logits.device)
                
                # 使用增强损失函数
                total_loss, local_loss_calc, global_loss_calc, balance_loss = self.enhanced_loss.stage2_3_loss(
                    local_logits, global_logits, target, 
                    personal_gradients=None, global_gradients=None,  # 简化实现
                    shared_features=shared_features, alpha=alpha
                )
                
                # 清除梯度
                shared_optimizer.zero_grad()
                server_optimizer.zero_grad()
                classifier_optimizer.zero_grad()
                
                # 反向传播
                total_loss.backward()
                
                # 应用梯度投影（简化版）
                # 在实际实现中，我们可以在这里应用梯度投影
                # 但为了保持代码简洁，暂时跳过复杂的梯度投影
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [p for n, p in client_model.named_parameters() if 'shared_base' in n], 
                    max_norm=0.5
                )
                
                # 更新参数
                shared_optimizer.step()
                server_optimizer.step()
                classifier_optimizer.step()
                
                # 更新统计信息
                stats['global_loss'] += global_loss.item()
                stats['local_loss'] += local_loss.item()
                stats['balance_loss'] += balance_loss.item()
                stats['total_loss'] += total_loss.item()
                stats['batch_count'] += 1
                
                _, pred = global_logits.max(1)
                stats['correct'] += pred.eq(target).sum().item()
                stats['total'] += target.size(0)
        
        # 计算平均值
        for key in ['global_loss', 'local_loss', 'balance_loss', 'total_loss']:
            if stats['batch_count'] > 0:
                stats[key] /= stats['batch_count']
        
        if stats['total'] > 0:
            stats['global_accuracy'] = 100.0 * stats['correct'] / stats['total']
        else:
            stats['global_accuracy'] = 0.0
        
        # 获取训练样本数
        train_samples = len(client.train_data.dataset) if hasattr(client.train_data, 'dataset') else stats['total']
        
        return {
            'global_loss': stats['global_loss'],
            'local_loss': stats['local_loss'],
            'balance_loss': stats['balance_loss'],
            'total_loss': stats['total_loss'],
            'global_accuracy': stats['global_accuracy'],
            'time_cost': time.time() - start_time,
            'train_samples': train_samples
        }

    def _evaluate_client(self, client, client_model, server_model, global_classifier):
        # 使用传入的参数而非self的模型，确保聚类专属模型被正确使用
        client_model.eval()
        server_model.eval()
        global_classifier.eval()

        # 关键：解包 DataLoader
        test_loader = self._unwrap_loader(client.test_data)

        total, correct_local, correct_global = 0, 0, 0
        loss_sum = 0.0

        with torch.no_grad():
            for data, target in test_loader:
                data   = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                local_logits, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)            # 使用传入的参数
                global_logits   = global_classifier(server_features)       # 使用传入的参数

                loss_sum += F.cross_entropy(global_logits, target, reduction="sum").item()
                pred_local  = local_logits.argmax(dim=1)
                pred_global = global_logits.argmax(dim=1)

                correct_local  += (pred_local  == target).sum().item()
                correct_global += (pred_global == target).sum().item()
                total += target.size(0)

        return {
            "test_loss":       loss_sum / max(1, total),
            "local_accuracy":  100.0 * correct_local  / max(1, total),
            "global_accuracy": 100.0 * correct_global / max(1, total),
        }

    def aggregate_client_shared_layers(self, shared_states, eval_results):
        """使用分层聚合器聚合客户端共享层"""
        if not shared_states:
            return {}
        
        return self.layered_aggregator.aggregate_shared_layers(shared_states, eval_results)

    def aggregate_server_models(self, eval_results=None):
        """使用分层聚合器聚合服务器模型"""
        if not self.cluster_server_models:
            return {}
        
        return self.layered_aggregator.aggregate_server_models(self.cluster_server_models, eval_results)

    def aggregate_global_classifiers(self, eval_results=None):
        """使用分层聚合器聚合全局分类器"""
        if not self.cluster_global_classifiers:
            return {}
        
        return self.layered_aggregator.aggregate_global_classifiers(self.cluster_global_classifiers, eval_results)

    def update_client_shared_layers(self, aggregated_shared_state):
        """更新所有客户端的共享层参数"""
        if not aggregated_shared_state:
            return False
        
        for client_id, model in self.client_models.items():
            for name, param in model.named_parameters():
                if 'shared_base' in name and name in aggregated_shared_state:
                    param.data.copy_(aggregated_shared_state[name])
        
        return True
    
    def update_server_models(self, aggregated_server_model, aggregated_global_classifier=None):
        """更新所有聚类的服务器模型和全局分类器"""
        updated = False
        
        # 更新服务器模型
        if aggregated_server_model:
            # 更新主服务器模型
            for name, param in self.server_model.named_parameters():
                if name in aggregated_server_model:
                    param.data.copy_(aggregated_server_model[name])
            
            # 更新所有聚类的服务器模型
            for cluster_id in self.cluster_server_models:
                self.cluster_server_models[cluster_id] = copy.deepcopy(self.server_model.state_dict())
            
            updated = True
        
        # 更新全局分类器
        if aggregated_global_classifier:
            # 更新主全局分类器
            for name, param in self.global_classifier.named_parameters():
                if name in aggregated_global_classifier:
                    param.data.copy_(aggregated_global_classifier[name])
            
            # 更新所有聚类的全局分类器
            for cluster_id in self.cluster_global_classifiers:
                self.cluster_global_classifiers[cluster_id] = copy.deepcopy(self.global_classifier.state_dict())
            
            updated = True
        
        return updated

    def select_clients_for_round(self, cluster_map, client_fraction=0.7, round_idx=0):
        """分层采样选择本轮参与训练的客户端"""
        selected_cluster_map = {}
        
        for cluster_id, client_ids in cluster_map.items():
            if len(client_ids) == 0:
                selected_cluster_map[cluster_id] = []
                continue
                
            # 计算该聚类中应选择的客户端数量
            k = max(1, int(len(client_ids) * client_fraction))
            
            if k >= len(client_ids):
                # 如果需要选择的数量大于等于总数，选择所有客户端
                selected_cluster_map[cluster_id] = client_ids
            else:
                # 基于tier和performance进行加权采样
                weights = []
                for cid in client_ids:
                    client = self.client_manager.get_client(cid)
                    if client:
                        # 基础权重：tier越高权重越大(tier=1最高，tier=4最低)
                        tier_weight = 5 - client.tier  # tier 1->4, tier 2->3, tier 3->2, tier 4->1
                        
                        # 性能权重：基于最近的速度和准确率
                        perf = self.client_performance[cid]
                        speed_weight = perf['speed_score'] 
                        acc_weight = perf['accuracy_ema'] + 0.1  # 加0.1避免零权重
                        
                        # 综合权重
                        total_weight = tier_weight * speed_weight * acc_weight
                        weights.append(max(total_weight, 0.01))  # 确保最小权重
                    else:
                        weights.append(0.01)
                
                # 归一化权重
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # 加权随机采样
                try:
                    selected_indices = np.random.choice(
                        len(client_ids), size=k, replace=False, p=weights
                    )
                    selected_cluster_map[cluster_id] = [client_ids[i] for i in selected_indices]
                except:
                    # 如果采样失败，退回到随机采样
                    selected_cluster_map[cluster_id] = np.random.choice(client_ids, size=k, replace=False).tolist()
        
        return selected_cluster_map
    
    def update_client_performance(self, client_id, training_time, accuracy):
        """更新客户端性能指标"""
        perf = self.client_performance[client_id]
        
        # 更新速度得分(EMA)
        speed_score = 1.0 / (training_time + 1e-6)  # 时间越短得分越高
        perf['speed_score'] = 0.8 * perf['speed_score'] + 0.2 * speed_score
        
        # 更新客户端性能指标
        perf['accuracy_ema'] = 0.8 * perf['accuracy_ema'] + 0.2 * accuracy

    def compute_prox_loss(self, local_model, global_model, mu=0.0):
        """计算FedProx正则化项"""
        if mu <= 0:
            return 0.0
        
        prox_loss = 0.0
        with torch.no_grad():
            for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
                prox_loss += torch.sum((local_param - global_param) ** 2)
        
        return mu / 2.0 * prox_loss


def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser(description='TierHFL: 分层异构联邦学习框架 (增强版本)')
    
    # 实验标识
    parser.add_argument('--running_name', default="TierHFL_Enhanced", type=str, help='实验名称')
    
    # 优化相关参数
    parser.add_argument('--lr', default=0.005, type=float, help='初始学习率')
    parser.add_argument('--lr_factor', default=0.9, type=float, help='学习率衰减因子')
    parser.add_argument('--wd', help='权重衰减参数', type=float, default=1e-4)
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='resnet56', help='使用的神经网络 (resnet56 或 resnet110)')
    
    # 数据加载和预处理相关参数
    parser.add_argument('--dataset', type=str, default='fashion_mnist', 
                       help='训练数据集 (cifar10, cifar100, fashion_mnist, cinic10)')
    parser.add_argument('--data_dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--partition_method', type=str, default='hetero', help='本地工作节点上数据集的划分方式')
    parser.add_argument('--partition_alpha', type=float, default=0.5, help='划分参数alpha')
    
    # 联邦学习相关参数
    parser.add_argument('--client_epoch', default=5, type=int, help='客户端本地训练轮数')
    parser.add_argument('--client_number', type=int, default=5, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=256, help='训练的输入批次大小')
    parser.add_argument('--rounds', default=100, type=int, help='联邦学习轮数')
    parser.add_argument('--n_clusters', default=3, type=int, help='客户端聚类数量')
    
    # TierHFL特有参数
    parser.add_argument('--init_alpha', default=0.6, type=float, help='初始本地与全局损失平衡因子')
    parser.add_argument('--min_alpha', default=0.4, type=float, help='最小本地与全局损失平衡因子')
    parser.add_argument('--init_lambda', default=0.15, type=float, help='初始特征对齐损失权重')
    parser.add_argument('--beta', default=0.3, type=float, help='聚合动量因子')
    
    # 分层采样参数
    parser.add_argument('--client_fraction', default=0.7, type=float, help='每轮参与训练的客户端比例(0-1)')
    parser.add_argument('--retier_interval', default=10, type=int, help='每隔多少轮重新分层/聚类')
    parser.add_argument('--mu', default=0.0, type=float, help='FedProx正则化系数(0表示关闭)')
    
    # 训练阶段参数
    parser.add_argument('--initial_feature_rounds', default=5, type=int, help='初始特征学习阶段轮数')
    parser.add_argument('--initial_phase_rounds', default=2, type=int, help='初始阶段轮数')
    parser.add_argument('--alternating_phase_rounds', default=0, type=int, help='交替训练阶段轮数(0表示自动计算)')
    parser.add_argument('--fine_tuning_phase_rounds', default=0, type=int, help='精细调整阶段轮数')

    parser.add_argument('--use_offline_wandb', default=0, type=int, help='是否使用离线wandb记录(1表示是)')
    parser.add_argument('--log_tag', default='', type=str, help='日志标签，用于区分不同实验')
    parser.add_argument('--target_accuracy', default=None, type=float, help='目标精度，达到后立即停止训练(如60.0表示60%)')
    parser.add_argument('--patience', default=15, type=int, help='早停耐心值，连续多少轮无改善后停止')

    parser.add_argument("--device", type=str, default="auto",
                    choices=["auto", "cuda", "cpu", "mps"],
                    help="auto 优先 cuda，其次 mps，最后 cpu")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度（仅在 cuda 时生效）")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader 的工作进程数（Kaggle 推荐 2~4）")
    
    # 训练阶段参数
    parser.add_argument("--server_first_warmup", action="store_true",
                        help="在初始阶段优先训练服务器模型")
    parser.add_argument("--warmup_rounds", type=int, default=2,
                        help="预热轮数")
    parser.add_argument("--disable_agg_rounds", type=int, default=0,
                        help="禁用聚合的轮数")

    
    args = parser.parse_args()
    return args

def setup_wandb(args):
    """仅负责初始化 wandb，不再动 logging 的 handler。"""
    logger = logging.getLogger("TierHFL")
    try:
        # 🔥 使用参数控制wandb模式和标签
        mode = "offline" if args.use_offline_wandb == 1 else "online"
        
        # 构建动态标签
        tags = [f"model_{args.model}", f"dataset_{args.dataset}",
               f"clients_{args.client_number}", f"partition_{args.partition_method}"]
        if args.log_tag:
            tags.append(args.log_tag)
            
        wandb.init(
            mode=mode,
            project="TierHFL_Enhanced",
            name=args.running_name,
            config=vars(args),
            tags=tags,
            group=f"{args.model}_{args.dataset}"
        )
        # 自定义面板
        wandb.define_metric("round")
        wandb.define_metric("global/*", step_metric="round")
        wandb.define_metric("local/*", step_metric="round")
        wandb.define_metric("client/*", step_metric="round")
        wandb.define_metric("time/*", step_metric="round")
        wandb.define_metric("params/*", step_metric="round")
        logger.info(f"wandb 初始化完成（{mode}）。")
    except Exception as e:
        logger.warning(f"wandb 初始化失败: {e}")
        try:
            wandb.init(mode="offline", project="TierHFL", name=args.running_name)
            logger.info("已切换到简化的 wandb 初始化（offline）。")
        except Exception:
            logger.warning("完全禁用 wandb。")


# --- Fallback: 本地 Fashion-MNIST 切分为多个客户端的简单实现 ---
def fallback_load_partition_data_fashion_mnist(dataset, data_dir, partition_method, partition_alpha,
                                              client_number, batch_size):
    import torchvision, torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    # Kaggle 无网时不要下载
    transform_train = T.Compose([T.RandomCrop(28, padding=2), T.ToTensor(), T.Normalize([0.2860],[0.3530])])
    transform_test  = T.Compose([T.ToTensor(), T.Normalize([0.2860],[0.3530])])

    # 尝试不下载；如本地无数据会抛错，再提示用 Kaggle Datasets
    try:
        trainset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=transform_train)
        testset  = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=False, transform=transform_test)
    except Exception:
        raise RuntimeError("本地未找到 Fashion-MNIST，Kaggle 无网状态请先在右侧 'Add data' 挂载 Fashion-MNIST 数据集或把数据拷到 ./data")

    n = len(trainset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    # 简单均分给各客户端（hetero/alpha 复杂切分此处先省略，能跑为先）
    splits = np.array_split(idx, client_number)

    train_data_local_dict, test_data_local_dict, train_data_local_num_dict = {}, {}, {}
    for cid in range(client_number):
        train_subset = Subset(trainset, splits[cid])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)
        test_loader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        train_data_local_dict[cid] = train_loader
        test_data_local_dict[cid] = test_loader
        train_data_local_num_dict[cid] = len(train_subset)

    train_data_num = len(trainset); test_data_num = len(testset); class_num = 10
    train_data_global = None; test_data_global = None
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num

def load_dataset(args):
    try:
        if args.dataset == "cifar10":
            from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
            data_loader = load_partition_data_cifar10
        elif args.dataset == "cifar100":
            from api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
            data_loader = load_partition_data_cifar100
        elif args.dataset == "fashion_mnist":
            from api.data_preprocessing.fashion_mnist.data_loader import load_partition_data_fashion_mnist
            data_loader = load_partition_data_fashion_mnist
        elif args.dataset == "cinic10":
            from api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
            data_loader = load_partition_data_cinic10
            args.data_dir = './data/cinic10/'
        else:
            from api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10
            data_loader = load_partition_data_cifar10

        if args.dataset == "cinic10":
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num, traindata_cls_counts = data_loader(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_number, args.batch_size)
            
            dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                       train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, traindata_cls_counts]
            
        else:
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_number, args.batch_size)
            
            dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                       train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    
    except Exception as e:
        print(f"[WARN] 正式数据加载器不可用，使用 fallback：{e}")
        if args.dataset == "fashion_mnist":
            train_data_num, test_data_num, train_data_global, test_data_global, \
            train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
            class_num = fallback_load_partition_data_fashion_mnist(args.dataset, args.data_dir, args.partition_method,
                                    args.partition_alpha, args.client_number, args.batch_size)
            
            dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
                       train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
        else:
            raise
    
    return dataset

def allocate_device_resources(client_number):
    resources = {}
    
    # 随机分配tier (1-4)
    tier_weights = [0.2, 0.3, 0.3, 0.2]  # tier 1-4的分布概率
    tiers = random.choices(range(1, 5), weights=tier_weights, k=client_number)
    
    # 为每个客户端分配资源
    for client_id in range(client_number):
        tier = tiers[client_id]
        
        # 根据tier分配计算能力
        if tier == 1:  # 高性能设备
            compute_power = random.uniform(0.8, 1.0)
            network_speed = random.choice([50, 100, 200])
            storage_capacity = random.choice([256, 512, 1024])
        elif tier == 2:  # 中高性能设备
            compute_power = random.uniform(0.6, 0.8)
            network_speed = random.choice([30, 50, 100])
            storage_capacity = random.choice([128, 256, 512])
        elif tier == 3:  # 中低性能设备
            compute_power = random.uniform(0.3, 0.6)
            network_speed = random.choice([20, 30, 50])
            storage_capacity = random.choice([64, 128, 256])
        else:  # tier 4, 低性能设备
            compute_power = random.uniform(0.1, 0.3)
            network_speed = random.choice([5, 10, 20])
            storage_capacity = random.choice([16, 32, 64])
        
        # 存储资源信息
        resources[client_id] = {
            "tier": tier,
            "compute_power": compute_power,
            "network_speed": network_speed,
            "storage_capacity": storage_capacity
        }
    
    return resources

def print_cluster_info(cluster_map, client_resources, logger):
    """打印聚类信息详情"""
    logger.info("===== 聚类分布情况 =====")
    for cluster_id, client_ids in cluster_map.items():
        client_tiers = [client_resources[client_id]['tier'] for client_id in client_ids]
        avg_tier = sum(client_tiers) / len(client_tiers) if client_tiers else 0
        tier_distribution = {}
        for tier in client_tiers:
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1
            
        logger.info(f"聚类 {cluster_id}: {len(client_ids)}个客户端")
        logger.info(f"  - 客户端ID: {client_ids}")
        logger.info(f"  - 平均Tier: {avg_tier:.2f}")
        logger.info(f"  - Tier分布: {tier_distribution}")
        
        # 计算客户端资源异质性
        if client_ids:
            compute_powers = [client_resources[cid]['compute_power'] for cid in client_ids]
            network_speeds = [client_resources[cid]['network_speed'] for cid in client_ids]
            
            logger.info(f"  - 计算能力: 平均={sum(compute_powers)/len(compute_powers):.2f}, "
                       f"最小={min(compute_powers):.2f}, 最大={max(compute_powers):.2f}")
            logger.info(f"  - 网络速度: 平均={sum(network_speeds)/len(network_speeds):.2f}, "
                       f"最小={min(network_speeds)}, 最大={max(network_speeds)}")
    
    # 计算全局聚类指标
    all_clients = sum(len(clients) for clients in cluster_map.values())
    logger.info(f"总计: {len(cluster_map)}个聚类, {all_clients}个客户端")

def load_global_test_set(args):
    """创建全局IID测试集用于评估泛化性能"""
    if args.dataset == "cifar10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.49139968, 0.48215827, 0.44653124], 
                                [0.24703233, 0.24348505, 0.26158768])
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cifar100":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                [0.2675, 0.2565, 0.2761])
        ])
        
        testset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "fashion_mnist":
        # 新增Fashion-MNIST支持
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.2860], [0.3530])
        ])
        
        testset = torchvision.datasets.FashionMNIST(
            root=args.data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    
    elif args.dataset == "cinic10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.47889522, 0.47227842, 0.43047404],
                                [0.24205776, 0.23828046, 0.25874835])
        ])
        
        # 使用存储在args.data_dir/cinic10/test目录下的CINIC10测试集
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(args.data_dir, 'cinic10', 'test'),
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        return test_loader
    else:
        # 默认返回CIFAR10
        raise ValueError(f"Unsupported dataset: {args.dataset}")

def evaluate_global_model_multi_client(client_models, server_model, global_classifier, global_test_loader, device, num_eval_clients=3):
    """🔥 使用多个客户端进行全局模型评估，提高评估稳健性"""
    # 随机选择多个客户端进行评估，避免固定选择前几个造成偏差
    client_ids = list(client_models.keys())
    
    if len(client_ids) <= num_eval_clients:
        eval_client_ids = client_ids  # 如果客户端数量不多，全部评估
    else:
        # 随机采样，确保评估的多样性
        import random
        eval_client_ids = random.sample(client_ids, num_eval_clients)
    
    all_accuracies = []
    
    for client_id in eval_client_ids:
        client_model = client_models[client_id]
        accuracy = evaluate_global_model(client_model, server_model, global_classifier, global_test_loader, device)
        all_accuracies.append(accuracy)
        print(f"📈 客户端{client_id}评估准确率: {accuracy:.2f}%")
    
    # 返回平均准确率
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
    print(f"📈 多客户端平均评估准确率: {avg_accuracy:.2f}%")
    
    return avg_accuracy

def evaluate_global_model(client_model, server_model, global_classifier, global_test_loader, device):
    """评估全局模型在全局测试集上的性能 - 修复版"""
    # 确保所有模型都在正确的设备上
    client_model = client_model.to(device)
    server_model = server_model.to(device)
    global_classifier = global_classifier.to(device)
    
    # 设置为评估模式
    client_model.eval()
    server_model.eval()
    global_classifier.eval()
    
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in global_test_loader:
            # 移到设备
            data, target = data.to(device), target.to(device)
            
            try:
                # 完整的前向传播
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                logits = global_classifier(server_features)
                
                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # 记录预测和目标，用于后续分析
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            except Exception as e:
                print(f"评估中出现错误: {str(e)}")
                continue
    
    accuracy = 100.0 * correct / max(1, total)
    
    return accuracy

class ModelFeatureClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.clustering_history = []
    
    def cluster_clients(self, client_models, client_ids, eval_dataset=None, device='cuda'):
        """基于模型特征的聚类方法，考虑服务器不能获取原始数据的限制"""
        clusters = {i: [] for i in range(self.num_clusters)}
        
        # 提取模型特征 - 使用共享层参数而非原始数据
        client_features = {}
        feature_dims = []
        
        # 第一步：提取特征
        for client_id in client_ids:
            if client_id in client_models:
                model = client_models[client_id]
                features = []
                
                # 只提取共享层参数作为特征
                for name, param in model.named_parameters():
                    if 'shared_base' in name and 'weight' in name:
                        # 提取统计信息而非原始参数
                        param_data = param.detach().cpu()
                        # 只收集标量特征，避免形状不一致
                        features.extend([
                            param_data.mean().item(),
                            param_data.std().item(),
                            param_data.abs().max().item(),
                            (param_data > 0).float().mean().item()  # 正值比例
                        ])
                
                if features:
                    # 确保features是一维数组
                    features_array = np.array(features, dtype=np.float32)
                    client_features[client_id] = features_array
                    feature_dims.append(len(features_array))
        
        # 检查所有特征向量的维度是否一致
        if feature_dims and len(set(feature_dims)) > 1:
            # 如果维度不一致，找出最常见的维度
            from collections import Counter
            dim_counter = Counter(feature_dims)
            common_dim = dim_counter.most_common(1)[0][0]
            
            print(f"发现不同维度的特征向量: {dict(dim_counter)}，使用最常见维度: {common_dim}")
            
            # 处理维度不一致的特征向量
            for client_id in list(client_features.keys()):
                feat = client_features[client_id]
                if len(feat) != common_dim:
                    if len(feat) < common_dim:
                        # 如果特征太短，使用填充
                        client_features[client_id] = np.pad(feat, (0, common_dim - len(feat)), 'constant')
                    else:
                        # 如果特征太长，进行裁剪
                        client_features[client_id] = feat[:common_dim]
        
        # 尝试K-means聚类
        if len(client_features) >= self.num_clusters:
            try:
                from sklearn.cluster import KMeans
                # 转换为矩阵
                feature_client_ids = list(client_features.keys())
                features_matrix = np.vstack([client_features[cid] for cid in feature_client_ids])
                
                # 标准化特征
                mean = np.mean(features_matrix, axis=0)
                std = np.std(features_matrix, axis=0) + 1e-8
                features_matrix = (features_matrix - mean) / std
                
                # 执行K-means
                kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
                kmeans.fit(features_matrix)
                
                # 构建聚类映射
                for i, label in enumerate(kmeans.labels_):
                    client_id = feature_client_ids[i]
                    clusters[label].append(client_id)
                
                # 处理没有特征的客户端 - 平均分配
                remaining_clients = [cid for cid in client_ids if cid not in client_features]
                for i, client_id in enumerate(remaining_clients):
                    target_cluster = i % self.num_clusters
                    clusters[target_cluster].append(client_id)
                    
            except Exception as e:
                print(f"K-means聚类失败: {str(e)}，使用备选方案")
                # 备用方案 - 均匀分配
                for i, client_id in enumerate(client_ids):
                    cluster_idx = i % self.num_clusters
                    clusters[cluster_idx].append(client_id)
        else:
            # 备用方案：均匀分配
            for i, client_id in enumerate(client_ids):
                cluster_idx = i % self.num_clusters
                clusters[cluster_idx].append(client_id)
        
        # 记录聚类结果
        self.clustering_history.append({
            'timestamp': time.time(),
            'clusters': copy.deepcopy(clusters),
            'num_clients': len(client_ids)
        })
            
        return clusters

# 主函数
def main():
    """主函数：增强版TierHFL实现"""
    # 解析命令行参数
    args = parse_arguments()

    # 🔥 训练阶段轮数自动计算（如果未指定）
    if args.alternating_phase_rounds == 0:
        args.alternating_phase_rounds = args.rounds - args.initial_phase_rounds - args.fine_tuning_phase_rounds
        args.alternating_phase_rounds = max(1, args.alternating_phase_rounds)  # 确保至少1轮

    log_file = setup_logging(run_name=getattr(args, "running_name", "run"))

    import logging
    logger = logging.getLogger("TierHFL")
    setup_wandb(args)

    logger.info("初始化TierHFL: 增强版本 - 集成梯度投影和分层聚合")

    # 设备
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # 提升卷积性能
    logging.info(f"使用设备: {device}")

    pin_mem = (str(device) == "cuda")

    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset}")
    dataset = load_dataset(args)
    
    # 获取数据集信息
    if args.dataset != "cinic10":
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
    else:
        train_data_num, test_data_num, _, _, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, _ = dataset
    
    # 加载全局测试集
    logger.info("加载全局IID测试集用于评估泛化性能...")
    global_test_loader = load_global_test_set(args)
    
    # 分配客户端资源
    logger.info(f"为 {args.client_number} 个客户端分配异构资源...")
    client_resources = allocate_device_resources(args.client_number)
    
    # 创建客户端管理器
    logger.info("创建客户端管理器...")
    client_manager = TierHFLClientManager()
    
    # === [MARK-CLIENT-DATALOADERS] 每个客户端先建 DataLoader 再注册 ===
    from torch.utils.data import DataLoader

    for client_id in range(args.client_number):
        resource = client_resources[client_id]
        tier = resource["tier"]

        # 1) 直接使用预处理好的 DataLoader（已经在数据预处理阶段创建）
        train_loader = train_data_local_dict[client_id]
        test_loader = test_data_local_dict[client_id]

        # 2) 把 DataLoader 传给客户端管理器
        client = client_manager.add_client(
            client_id=client_id,
            tier=tier,
            train_data=train_loader,
            test_data=test_loader,
            device=device,
            lr=args.lr,
            local_epochs=args.client_epoch
        )
        client.wd = args.wd
    # === [MARK-CLIENT-DATALOADERS] END ===



    # 确定输入通道数
    input_channels = 1 if args.dataset == "fashion_mnist" else 3

    # 创建客户端模型
    logger.info("创建双路径客户端模型...")
    client_models = {}
    for client_id, resource in client_resources.items():
        client_models[client_id] = TierAwareClientModel(
            num_classes=class_num,
            input_channels=(1 if args.dataset in ["mnist", "fashion_mnist"] else 3)
        )
        
    # 创建服务器特征提取模型
    logger.info("创建服务器特征提取模型...")
    server_model = EnhancedServerModel(
        model_type=args.model,
        feature_dim=384,  # 🔥 从256提升到384，进一步提升CIFAR-100表征能力
        input_channels=input_channels  # 添加输入通道参数
    ).to(device)
    
    # 创建全局分类器
    logger.info("创建全局分类器...")
    global_classifier = ImprovedGlobalClassifier(
        feature_dim=384,  # 🔥 从256提升到384，匹配服务器特征维度
        num_classes=class_num
    ).to(device)
    
    # 创建客户端聚类器
    logger.info("创建数据分布聚类器...")
    clusterer = ModelFeatureClusterer(num_clusters=args.n_clusters)
    
    # 对客户端进行初始聚类
    logger.info("执行初始客户端聚类...")
    client_ids = list(range(args.client_number))
    cluster_map = clusterer.cluster_clients(
        client_models=client_models,
        client_ids=client_ids
    )
    
    # 打印初始聚类信息
    print_cluster_info(cluster_map, client_resources, logger)
    
    # 创建增强版串行训练器
    logger.info("创建增强版串行训练器...")
    trainer = EnhancedSerialTrainer(
        client_manager=client_manager,
        server_model=server_model,
        global_classifier=global_classifier,
        device=str(device),
        use_amp=args.amp
    )
    
    # 🔥 设置FedProx正则化系数
    trainer.mu = args.mu

    
    # 注册客户端模型
    trainer.register_client_models(client_models)
    
    # 设置训练环境
    trainer.setup_training(cluster_map=cluster_map)

    # 创建诊断监控器
    logger.info("创建诊断监控器...")
    diagnostic_monitor = EnhancedTierHFLDiagnosticMonitor(device='cpu')
    
    # 🔥 创建全局学习率调度器（跨轮Cosine调度）
    # 收集服务器模型和全局分类器的参数用于全局调度
    global_params = list(server_model.parameters()) + list(global_classifier.parameters())
    global_optimizer = torch.optim.SGD(
        global_params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    
    # Cosine学习率调度，带5轮warmup
    def cosine_lr_with_warmup(optimizer, current_round, total_rounds, warmup_rounds=5, eta_min=1e-4):
        if current_round < warmup_rounds:
            # Warmup阶段：线性增长到初始学习率
            lr_factor = (current_round + 1) / warmup_rounds
        else:
            # Cosine退火阶段
            progress = (current_round - warmup_rounds) / (total_rounds - warmup_rounds)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
            lr_factor = max(lr_factor, eta_min / args.lr)  # 确保不低于eta_min
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * lr_factor
        
        return args.lr * lr_factor
    
    # 开始训练循环
    logger.info(f"开始联邦学习训练 ({args.rounds} 轮)...")
    best_accuracy = 0.0
    prev_global_acc = 0.0
    
    # 🔥 早停参数
    patience = args.patience  # 使用参数配置的耐心值
    best_round = 0
    no_improve_count = 0
    target_accuracy = args.target_accuracy  # 目标精度
    
    # 在训练开始前进行初始验证
    initial_validation = validate_server_effectiveness(
        args, 
        client_models,
        server_model, 
        global_classifier,
        global_test_loader, 
        test_data_local_dict,
        device='cpu'
    )

    for round_idx in range(args.rounds):
        round_start_time = time.time()
        logger.info(f"===== 轮次 {round_idx+1}/{args.rounds} =====")
        
        # 添加训练阶段信息
        if round_idx < args.initial_phase_rounds:
            logger.info("当前处于初始特征学习阶段")
        elif round_idx < args.initial_phase_rounds + args.alternating_phase_rounds:
            logger.info("当前处于交替训练阶段")
        else:
            logger.info("当前处于精细调整阶段")
        
        # 执行训练 - 传递增强版诊断监控器
        train_results, eval_results, shared_states, training_time = trainer.execute_round(
            round_idx=round_idx, 
            total_rounds=args.rounds,
            args=args,
            diagnostic_monitor=diagnostic_monitor
        )
        
        # 聚合过程
        logger.info("使用分层聚合策略聚合模型...")
        aggregation_start_time = time.time()
        
        # 聚合客户端共享层
        aggregated_shared_state = trainer.aggregate_client_shared_layers(shared_states, eval_results)
        
        # 聚合服务器模型
        aggregated_server_model = trainer.aggregate_server_models(eval_results)

        # 聚合全局分类器
        aggregated_global_classifier = trainer.aggregate_global_classifiers(eval_results)
        
        # 更新模型
        logger.info("更新客户端共享层...")
        trainer.update_client_shared_layers(aggregated_shared_state)
        
        logger.info("更新服务器模型...")
        trainer.update_server_models(aggregated_server_model, aggregated_global_classifier)
        
        aggregation_time = time.time() - aggregation_start_time
        
        # 🔥 优化：使用多客户端评估全局模型，提高评估稳健性
        global_model_accuracy = evaluate_global_model_multi_client(
            client_models, server_model, global_classifier, 
            global_test_loader, device, num_eval_clients=3)
        
        # 计算平均准确率
        avg_local_acc = np.mean([result.get('local_accuracy', 0) for result in eval_results.values()])
        avg_global_acc = np.mean([result.get('global_accuracy', 0) for result in eval_results.values()])
        
        # 选择一个用于保存的样例客户端
        tier1_clients = [cid for cid, resource in client_resources.items() if resource.get("tier", 2) == 1]
        if len(tier1_clients) > 0:
            sample_client_id = int(tier1_clients[0])
        else:
            # 兜底：随便拿一个存在的客户端ID
            sample_client_id = int(list(client_models.keys())[0])
        
        # 更新最佳准确率
        is_best = global_model_accuracy > best_accuracy
        if is_best:
            best_accuracy = global_model_accuracy
            try:
                torch.save({
                    'client_model': client_models[sample_client_id].state_dict(),
                    'server_model': server_model.state_dict(),
                    'global_classifier': global_classifier.state_dict(),
                    'round': round_idx,
                    'accuracy': best_accuracy
                }, f"{args.running_name}_best_model.pth")
                logger.info(f"保存最佳模型，准确率: {best_accuracy:.2f}%")
            except Exception as e:
                logger.error(f"保存模型失败: {str(e)}")
        
        # 🔥 达标即停检查
        if target_accuracy is not None and global_model_accuracy >= target_accuracy:
            logger.info(f"达到目标精度 {target_accuracy:.1f}%! 当前精度: {global_model_accuracy:.2f}% (轮次 {round_idx+1})")
            logger.info(f"提前停止训练，最佳精度: {best_accuracy:.2f}%")
            break
        
        # 🔥 早停检查
        if global_model_accuracy > best_accuracy + 1e-4:  # 如果有显著改善
            best_accuracy = global_model_accuracy
            best_round = round_idx
            no_improve_count = 0
            logger.info(f"新的最佳准确率: {best_accuracy:.2f}% (轮次 {round_idx+1})")
        else:
            no_improve_count += 1
            logger.info(f"无改善计数: {no_improve_count}/{patience}")
        
        # 判断是否需要早停
        if no_improve_count >= patience:
            logger.info(f"早停触发! 已连续{patience}轮无改善")
            logger.info(f"最佳准确率: {best_accuracy:.2f}% (轮次 {best_round+1})")
            break
        
        # 计算轮次时间
        round_time = time.time() - round_start_time
        
        # 输出统计信息
        logger.info(f"轮次 {round_idx+1} 统计:")
        logger.info(f"本地平均准确率: {avg_local_acc:.2f}%, 全局平均准确率: {avg_global_acc:.2f}%")
        logger.info(f"全局模型在独立测试集上的准确率: {global_model_accuracy:.2f}%")
        logger.info(f"最佳准确率: {best_accuracy:.2f}%")
        logger.info(f"轮次总时间: {round_time:.2f}秒, 训练: {training_time:.2f}秒, 聚合: {aggregation_time:.2f}秒")
        
        # 记录准确率变化
        acc_change = global_model_accuracy - prev_global_acc
        prev_global_acc = global_model_accuracy
        logger.info(f"全局准确率变化: {acc_change:.2f}%")
        
        # === 增强版诊断报告 ===
        if round_idx % 3 == 0:  # 每3轮生成一次详细报告
            logger.info("=== 增强版诊断报告 ===")
            comprehensive_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            
            logger.info(f"系统健康状态: {comprehensive_report['overall_health'].upper()}")
            
            if comprehensive_report['critical_issues']:
                logger.error("严重问题:")
                for issue in comprehensive_report['critical_issues']:
                    logger.error(f"  🚨 {issue}")
            
            if comprehensive_report['warnings']:
                logger.warning("警告:")
                for warning in comprehensive_report['warnings']:
                    logger.warning(f"  ⚠️ {warning}")
            
            if comprehensive_report['recommendations']:
                logger.info("优化建议:")
                for rec in comprehensive_report['recommendations']:
                    logger.info(f"  💡 {rec}")
        
        # 导出增强版诊断指标到wandb
        try:
            diagnostic_monitor.export_metrics_to_wandb(wandb)
        except Exception as e:
            logger.error(f"导出增强诊断指标失败: {str(e)}")
        
        # 记录到wandb（添加健康状态指标）
        try:
            # 获取当前健康状态
            current_report = diagnostic_monitor.comprehensive_diagnostic_report(round_idx)
            health_score = {'good': 1.0, 'warning': 0.7, 'poor': 0.4, 'critical': 0.0, 'error': 0.0}.get(
                current_report['overall_health'], 0.5)
            
            metrics = {
                "round": round_idx + 1,
                "global/test_accuracy": global_model_accuracy,
                "global/best_accuracy": best_accuracy,
                "local/avg_accuracy": avg_local_acc,
                "global/avg_accuracy": avg_global_acc,
                "time/round_seconds": round_time,
                "time/training_seconds": training_time,
                "time/aggregation_seconds": aggregation_time,
                "global/accuracy_change": acc_change,
                "diagnostic/system_health_score": health_score,
                "training/phase": 1 if round_idx < args.initial_phase_rounds else 
                                (2 if round_idx < args.initial_phase_rounds + args.alternating_phase_rounds else 3)
            }
            
            # 添加特征平衡和损失组件指标
            if train_results:
                avg_balance_loss = np.mean([result.get('balance_loss', 0) for result in train_results.values()])
                metrics["training/avg_balance_loss"] = avg_balance_loss
            
            wandb.log(metrics)
        except Exception as e:
            logger.error(f"记录wandb指标失败: {str(e)}")
        
        # 🔥 使用参数控制的重新聚类间隔
        if (round_idx + 1) % args.retier_interval == 0 and round_idx >= args.initial_phase_rounds:
            logger.info(f"重新进行客户端聚类（间隔：{args.retier_interval}轮）...")
            try:
                cluster_map = clusterer.cluster_clients(
                    client_models=client_models,
                    client_ids=client_ids,
                    eval_dataset=global_test_loader
                )
                trainer.setup_training(cluster_map=cluster_map)
                print_cluster_info(cluster_map, client_resources, logger)
            except Exception as e:
                logger.error(f"重新聚类失败: {str(e)}")
        
        # 🔥 全局Cosine学习率调度
        current_global_lr = cosine_lr_with_warmup(
            global_optimizer, round_idx, args.rounds, warmup_rounds=5, eta_min=1e-4
        )
        
        # 同步客户端学习率（可选择性地设置为全局学习率的一定比例）
        client_lr_ratio = 1.0  # 客户端与全局学习率的比例
        for client_id in range(args.client_number):
            client = client_manager.get_client(client_id)
            if client:
                client.lr = current_global_lr * client_lr_ratio
        
        if round_idx % 10 == 0:  # 每10轮记录一次学习率
            logger.info(f"轮次 {round_idx+1}: 全局学习率={current_global_lr:.6f}, 客户端学习率={current_global_lr * client_lr_ratio:.6f}")

        # 每隔5轮进行一次服务器有效性验证
        if (round_idx + 1) % 5 == 0:
            round_validation = validate_server_effectiveness(
                args, 
                client_models,
                server_model, 
                global_classifier,
                global_test_loader, 
                test_data_local_dict
            )
            
            # 🔥 计算客户端特征相似度（使用修复后的函数）
            try:
                from analyze.diagnostic_monitor import compute_client_feature_similarity_robust
                similarity = compute_client_feature_similarity_robust(
                    client_models, server_model, global_test_loader, device
                )
                logger.info(f"轮次 {round_idx+1} 客户端特征平均相似度: {similarity:.4f}")
                
                # 记录到wandb
                if wandb.run:
                    wandb.log({"diagnostic/feature_similarity": similarity, "round": round_idx + 1})
                    
            except Exception as e:
                logger.warning(f"特征相似度计算失败: {str(e)}")
            
            # 记录验证结果
            try:
                wandb.log({
                    "round": round_idx + 1,
                    "validation/feature_quality": round_validation['feature_quality'],
                    "validation/heterogeneity_adaptation": round_validation['heterogeneity_adaptation'],
                    "validation/simple_classifier_acc": round_validation['simple_classifier_acc']
                })
            except Exception as e:
                logger.error(f"记录wandb验证指标失败: {str(e)}")
    
    # 训练完成后的最终报告
    logger.info("=== 最终诊断报告 ===")
    final_report = diagnostic_monitor.comprehensive_diagnostic_report(args.rounds - 1)
    
    logger.info(f"训练完成! 最佳准确率: {best_accuracy:.2f}%")
    logger.info(f"最终系统健康状态: {final_report['overall_health'].upper()}")
    
    if final_report['critical_issues']:
        logger.error("遗留的严重问题:")
        for issue in final_report['critical_issues']:
            logger.error(f"  - {issue}")
    
    if final_report['recommendations']:
        logger.info("后续优化建议:")
        for rec in final_report['recommendations']:
            logger.info(f"  - {rec}")
    
    # 关闭wandb
    try:
        wandb.finish()
    except:
        pass

if __name__ == "__main__":
    main()