import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import copy
import numpy as np
import math

# 混合损失 - 平衡个性化和全局性能
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha  # 平衡因子
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, local_logits, global_logits, target):
        local_loss = self.criterion(local_logits, target)
        global_loss = self.criterion(global_logits, target)
        return self.alpha * local_loss + (1 - self.alpha) * global_loss, local_loss, global_loss
    
    def update_alpha(self, alpha):
        """更新平衡因子"""
        self.alpha = alpha

# 特征对齐损失 - 处理不同维度的特征
class EnhancedFeatureAlignmentLoss(nn.Module):
    def __init__(self):
        super(EnhancedFeatureAlignmentLoss, self).__init__()
        
    def forward(self, client_features, server_features, round_idx=0):
        """改进的特征对齐损失计算"""
        # 添加调试模式
        debug_mode = hasattr(self, '_debug_client_id') and self._debug_client_id == 6
        
        if debug_mode:
            print(f"\n[Feature Loss DEBUG] 客户端特征形状: {client_features.shape}")
            print(f"[Feature Loss DEBUG] 服务器特征形状: {server_features.shape}")
        
        # 确保特征是4D或2D张量
        if len(client_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = client_features.size(0)
            client_pooled = F.adaptive_avg_pool2d(client_features, (1, 1))
            client_features = client_pooled.view(batch_size, -1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 池化后客户端特征形状: {client_features.shape}")
        
        if len(server_features.shape) == 4:  # 4D: [B, C, H, W]
            batch_size = server_features.size(0)
            server_pooled = F.adaptive_avg_pool2d(server_features, (1, 1))
            server_features = server_pooled.view(batch_size, -1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 池化后服务器特征形状: {server_features.shape}")
        
        # 检查特征维度一致性 - 如果不一致说明模型架构有问题
        if client_features.size(1) != server_features.size(1):
            if debug_mode:
                print(f"[Feature Loss DEBUG] 特征维度不匹配! 客户端: {client_features.size(1)}, 服务器: {server_features.size(1)}")
            
            # 🔥 不再裁切，而是报警告并返回零损失，避免信息损失
            logging.warning(f"特征维度不匹配：客户端{client_features.size(1)} vs 服务器{server_features.size(1)}，跳过特征对齐损失计算")
            return torch.tensor(0.0, device=client_features.device, requires_grad=True)
        
        # 标准化特征向量并检测异常值
        try:
            client_norm = F.normalize(client_features, dim=1)
            server_norm = F.normalize(server_features, dim=1)
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 客户端归一化后是否有NaN: {torch.isnan(client_norm).any().item()}")
                print(f"[Feature Loss DEBUG] 服务器归一化后是否有NaN: {torch.isnan(server_norm).any().item()}")
            
            # 余弦相似度
            cosine_sim = torch.mean(torch.sum(client_norm * server_norm, dim=1))
            cosine_loss = 1.0 - cosine_sim
            
            if debug_mode:
                print(f"[Feature Loss DEBUG] 余弦相似度: {cosine_sim.item():.4f}")
                print(f"[Feature Loss DEBUG] 特征对齐损失: {cosine_loss.item():.4f}")
        except Exception as e:
            if debug_mode:
                print(f"[Feature Loss DEBUG] 计算特征对齐损失出错: {str(e)}")
            # 出错时返回一个默认损失值
            return torch.tensor(1.0, device=client_features.device)
        
        # 随训练轮次渐进增强特征对齐强度
        alignment_weight = min(0.8, 0.2 + round_idx/100)
        
        return cosine_loss * alignment_weight

class TierHFLClient:
    """TierHFL客户端类"""
    def __init__(self, client_id, tier, train_data, test_data, device='cuda', 
                 lr=0.001, local_epochs=1):
        self.client_id = client_id
        self.tier = tier
        self.train_data = train_data
        self.test_data = test_data
        # [MARK-CLIENT-DEVICE] normalize device
        if not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.lr = lr
        self.local_epochs = local_epochs
        
        # 模型引用 - 将在训练时设置
        self.model = None
        
        # 训练统计信息
        self.stats = {
            'train_loss': [],
            'train_acc': [],
            'local_loss': [],
            'global_loss': [],
            'feature_loss': []
        }
        
        # 动态参数
        self.alpha = 0.5  # 本地损失权重
        self.lambda_feature = 0.1  # 特征对齐损失权重
        
        # AMP 支持
        self.use_amp = (isinstance(device, torch.device) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
    
    def update_learning_rate(self, lr_factor=0.85):
        """更新学习率"""
        self.lr *= lr_factor
        return self.lr
    
    def train_personalized_layers(self, round_idx=0, total_rounds=100):
        """只训练个性化层，并收集共享层特征"""
        if self.model is None:
            raise ValueError("客户端模型未设置")
        
        self.model.train()
        
        # 冻结共享层
        for name, param in self.model.named_parameters():
            if 'shared_base' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 创建优化器 - 只优化个性化层
        optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'shared_base' not in n],
            lr=self.lr
        )
        
        # 可选：创建学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=3, verbose=False
        )
        
        # 统计信息
        stats = {
            'local_loss': 0.0,
            'correct': 0,
            'total': 0,
            'batch_count': 0
        }
        
        # 保存的特征数据
        features_data = []
        
        # 记录开始时间
        start_time = time.time()
        
        # 计算进度因子 - 用于动态调整超参数
        progress = round_idx / max(1, total_rounds)
        
        # 训练循环
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (data, target) in enumerate(self.train_data):
                # 移至设备 - 添加 non_blocking=True
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 清除梯度
                optimizer.zero_grad(set_to_none=True)
                
                # 前向传播 - 添加 AMP 支持
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    local_logits, shared_features, personal_features = self.model(data)
                    local_loss = F.cross_entropy(local_logits, target)
                
                # 反向传播 - 添加 AMP 支持
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(local_loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    local_loss.backward()
                    optimizer.step()
                
                # 更新统计信息
                epoch_loss += local_loss.item()
                stats['local_loss'] += local_loss.item()
                stats['batch_count'] += 1
                
                # 计算准确率
                _, pred = local_logits.max(1)
                batch_correct = pred.eq(target).sum().item()
                batch_total = target.size(0)
                
                epoch_correct += batch_correct
                epoch_total += batch_total
                stats['correct'] += batch_correct
                stats['total'] += batch_total
                
                # 保存特征数据
                features_data.append({
                    'shared_features': shared_features.detach(),
                    'personal_features': personal_features.detach(),
                    'targets': target.clone(),
                    'local_loss': local_loss.item()
                })
            
            # 每轮结束后更新学习率
            epoch_acc = 100.0 * epoch_correct / max(1, epoch_total)
            scheduler.step(epoch_acc)
        
        # 计算平均值和总体准确率
        avg_local_loss = stats['local_loss'] / max(1, stats['batch_count'])
        avg_local_acc = 100.0 * stats['correct'] / max(1, stats['total'])
        
        # 更新统计信息
        self.stats['local_loss'].append(avg_local_loss)
        self.stats['train_acc'].append(avg_local_acc)
        
        return {
            'local_loss': avg_local_loss,
            'local_accuracy': avg_local_acc,
            'time_cost': time.time() - start_time,
            'features_data': features_data
        }
    
    def evaluate(self, server_model, global_classifier):
        """评估客户端模型"""
        if self.model is None:
            raise ValueError("客户端模型未设置")

        # 关键：把服务端与全局头搬到同一个 device
        server_model       = server_model.to(self.device)
        global_classifier  = global_classifier.to(self.device)
        
        # 设置为评估模式
        self.model.eval()
        server_model.eval()
        global_classifier.eval()
        
        # 统计信息
        local_correct = 0
        global_correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_data:
                # 移到设备 - 添加 non_blocking=True
                # [MARK-NONBLOCK-EVAL]
                data   = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # 前向传播 - 添加 AMP 支持
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    local_logits, shared_features, _ = self.model(data)
                    server_features = server_model(shared_features)
                    global_logits = global_classifier(server_features)
                
                # 计算准确率
                _, local_pred = local_logits.max(1)
                _, global_pred = global_logits.max(1)
                
                local_correct += local_pred.eq(target).sum().item()
                global_correct += global_pred.eq(target).sum().item()
                total += target.size(0)
        
        # 计算准确率
        local_accuracy = 100.0 * local_correct / max(1, total)
        global_accuracy = 100.0 * global_correct / max(1, total)
        
        return {
            'local_accuracy': local_accuracy,
            'global_accuracy': global_accuracy,
            'total_samples': total
        }
    
    def apply_shared_layer_gradients(self, shared_grads):
        """应用服务器计算的共享层梯度"""
        if self.model is None:
            raise ValueError("客户端模型未设置")
            
        # 创建优化器
        shared_optimizer = torch.optim.Adam(
            [p for n, p in self.model.named_parameters() if 'shared_base' in n],
            lr=self.lr * 0.5  # 共享层使用较小学习率
        )
        
        # 应用梯度
        for name, param in self.model.named_parameters():
            if 'shared_base' in name and name in shared_grads:
                if param.grad is None:
                    param.grad = shared_grads[name].clone()
                else:
                    param.grad.copy_(shared_grads[name])
        
        # 更新参数
        shared_optimizer.step()
        shared_optimizer.zero_grad(set_to_none=True)
        
        return True
    
    def update_alpha(self, alpha):
        """更新本地和全局损失的平衡因子"""
        self.alpha = alpha
    
    def update_lambda_feature(self, lambda_feature):
        """更新特征对齐损失权重"""
        self.lambda_feature = lambda_feature

# 客户端管理器
class TierHFLClientManager:
    def __init__(self):
        self.clients = {}
        self.default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # 在客户端管理器中添加监控方法
    def add_client(self, client_id, tier, train_data, test_data, device=None, lr=0.001, local_epochs=1):
        """添加客户端"""
        device = device or self.default_device
        
        client = TierHFLClient(
            client_id=client_id,
            tier=tier,
            train_data=train_data,
            test_data=test_data,
            device=device,
            lr=lr,
            local_epochs=local_epochs
        )
        
        # 针对客户端6添加特殊监控
        if client_id == 6:
            print(f"\n[CLIENT MANAGER] 注册客户端6 - Tier: {tier}")
            print(f"[CLIENT MANAGER] 客户端6训练集样本数: {len(train_data.dataset)}")
            print(f"[CLIENT MANAGER] 客户端6测试集样本数: {len(test_data.dataset)}")
            
            # 检查数据集分布
            try:
                # 获取前5个样本的标签
                print("[CLIENT MANAGER] 分析客户端6数据集...")
                sample_labels = []
                for i, (_, labels) in enumerate(train_data):
                    sample_labels.extend(labels.tolist())
                    if i >= 2:  # 只检查前几个批次
                        break
                
                # 统计标签分布
                label_counts = {}
                for label in sample_labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                    
                print(f"[CLIENT MANAGER] 客户端6训练集标签分布(部分): {label_counts}")
            except Exception as e:
                print(f"[CLIENT MANAGER] 分析客户端6数据时出错: {str(e)}")
        
        self.clients[client_id] = client
        return client
    
    def get_client(self, client_id):
        """获取客户端"""
        return self.clients.get(client_id)
    
    def update_client_tier(self, client_id, new_tier):
        """更新客户端的tier级别"""
        if client_id in self.clients:
            self.clients[client_id].tier = new_tier
            return True
        return False
    
    def update_client_alpha(self, client_id, alpha):
        """更新客户端的alpha值"""
        if client_id in self.clients:
            self.clients[client_id].update_alpha(alpha)
            return True
        return False
    
    def update_client_feature_lambda(self, client_id, lambda_feature):
        """更新客户端的特征对齐损失权重"""
        if client_id in self.clients:
            self.clients[client_id].update_lambda_feature(lambda_feature)
            return True
        return False
    
    def update_all_clients_alpha(self, alpha):
        """更新所有客户端的alpha值"""
        for client in self.clients.values():
            client.update_alpha(alpha)
    
    def update_all_clients_feature_lambda(self, lambda_feature):
        """更新所有客户端的特征对齐损失权重"""
        for client in self.clients.values():
            client.update_lambda_feature(lambda_feature)
