import torch
import torch.nn.functional as F
import numpy as np
import logging
import copy
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

class EnhancedTierHFLDiagnosticMonitor:
    """增强版TierHFL诊断监控器 - 修复错误并添加核心问题排查"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.gradient_history = []
        self.feature_history = []
        self.loss_history = []
        self.model_stability_history = []
        
        # 监控指标
        self.metrics = {
            'gradient_conflict': [],
            'gradient_norm_ratio': [],
            'feature_similarity': [],
            'feature_diversity': [],
            'feature_stability': [],
            'loss_components': [],
            'model_parameter_change': [],
            'classification_confidence': [],
            'classifier_collapse': [],
            'weight_distribution': [],
            'shared_layer_quality': [],
            'aggregation_analysis': []
        }
        
        # 设置日志
        self.logger = logging.getLogger("EnhancedDiagnosticMonitor")
        
        # 核心问题追踪
        self.collapse_detected = False
        self.previous_predictions = None
        
    def analyze_gradient_conflict_fixed(self, client_model, global_loss, local_loss, round_idx, client_id):
        """修复版梯度冲突分析"""
        try:
            # 检查损失是否需要梯度
            if not isinstance(global_loss, torch.Tensor) or not isinstance(local_loss, torch.Tensor):
                self.logger.debug(f"轮次{round_idx} 客户端{client_id}: 损失不是张量类型")
                return {}
            
            # 确保损失需要梯度
            if not global_loss.requires_grad:
                self.logger.debug(f"轮次{round_idx} 客户端{client_id}: 全局损失不需要梯度")
                return {}
                
            # 获取共享层参数
            shared_params = []
            for name, param in client_model.named_parameters():
                if 'shared_base' in name and param.requires_grad and param.grad is None:
                    shared_params.append(param)
            
            if not shared_params:
                self.logger.debug(f"轮次{round_idx} 客户端{client_id}: 没有找到可用的共享层参数")
                return {}
            
            # 计算全局损失梯度
            try:
                global_grads = torch.autograd.grad(
                    global_loss, shared_params, retain_graph=True, 
                    create_graph=False, allow_unused=True
                )
            except Exception as e:
                self.logger.debug(f"全局损失梯度计算失败: {str(e)}")
                return {}
            
            # 计算本地损失梯度 - 修复：处理frozen参数
            if hasattr(local_loss, 'requires_grad') and local_loss.requires_grad:
                try:
                    local_grads = torch.autograd.grad(
                        local_loss, shared_params, retain_graph=True, 
                        create_graph=False, allow_unused=True
                    )
                except Exception as e:
                    self.logger.debug(f"本地损失梯度计算失败: {str(e)}")
                    # 如果本地损失梯度计算失败，使用零梯度作为替代
                    local_grads = [torch.zeros_like(p) for p in shared_params]
            else:
                # 本地损失不需要梯度，使用零梯度
                local_grads = [torch.zeros_like(p) for p in shared_params]
            
            # 计算梯度相似度
            similarities = []
            norm_ratios = []
            
            for g_grad, l_grad in zip(global_grads, local_grads):
                if g_grad is not None and l_grad is not None:
                    g_flat = g_grad.flatten()
                    l_flat = l_grad.flatten()
                    
                    g_norm = torch.norm(g_flat)
                    l_norm = torch.norm(l_flat)
                    
                    if g_norm > 1e-8 and l_norm > 1e-8:
                        # 计算余弦相似度
                        cos_sim = F.cosine_similarity(g_flat.unsqueeze(0), l_flat.unsqueeze(0))
                        similarities.append(cos_sim.item())
                        
                        # 计算范数比率
                        norm_ratios.append((g_norm / l_norm).item())
            
            if not similarities:
                return {}
            
            avg_similarity = np.mean(similarities)
            avg_norm_ratio = np.mean(norm_ratios)
            
            result = {
                'round': round_idx,
                'client_id': client_id,
                'avg_cosine_similarity': avg_similarity,
                'avg_norm_ratio': avg_norm_ratio,
                'conflict_level': self._classify_conflict_level(avg_similarity),
                'valid_grad_pairs': len(similarities)
            }
            
            self.metrics['gradient_conflict'].append(result)
            
            if avg_similarity < -0.1:
                self.logger.warning(f"轮次{round_idx} 客户端{client_id}: 严重梯度冲突! 相似度: {avg_similarity:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"梯度冲突分析失败: {str(e)}")
            return {}
    
    def monitor_model_stability_fixed(self, model_state_dict, round_idx, model_type="client"):
        """修复版模型稳定性监控"""
        try:
            if not hasattr(self, f'previous_{model_type}_state'):
                setattr(self, f'previous_{model_type}_state', copy.deepcopy(model_state_dict))
                return {}
            
            previous_state = getattr(self, f'previous_{model_type}_state')
            
            param_changes = []
            param_norms = []
            
            for name, param in model_state_dict.items():
                if name in previous_state:
                    # 确保参数是浮点型张量
                    if not param.dtype.is_floating_point:
                        continue
                    
                    prev_param = previous_state[name]
                    if not prev_param.dtype.is_floating_point:
                        continue
                    
                    # 计算参数变化
                    try:
                        change = torch.norm(param.float() - prev_param.float()).item()
                        param_changes.append(change)
                        
                        # 参数范数
                        norm = torch.norm(param.float()).item()
                        param_norms.append(norm)
                    except Exception as e:
                        self.logger.debug(f"参数{name}稳定性计算失败: {str(e)}")
                        continue
            
            if not param_changes:
                return {}
            
            avg_change = np.mean(param_changes)
            max_change = np.max(param_changes)
            avg_norm = np.mean(param_norms)
            
            stability_analysis = {
                'round': round_idx,
                'model_type': model_type,
                'avg_parameter_change': avg_change,
                'max_parameter_change': max_change, 
                'avg_parameter_norm': avg_norm,
                'stability_score': self._compute_stability_score(avg_change, avg_norm),
                'param_count': len(param_changes)
            }
            
            self.metrics['model_parameter_change'].append(stability_analysis)
            
            # 更新历史状态
            setattr(self, f'previous_{model_type}_state', copy.deepcopy(model_state_dict))
            
            if stability_analysis['stability_score'] < 0.2:
                self.logger.warning(f"轮次{round_idx} {model_type}模型: 参数变化过大, 稳定性得分: {stability_analysis['stability_score']:.4f}")
            
            return stability_analysis
            
        except Exception as e:
            self.logger.error(f"模型稳定性监控失败: {str(e)}")
            return {}
    
    def detect_classifier_collapse(self, logits, targets, round_idx, client_id, path_type="global"):
        """检测分类器崩溃"""
        try:
            with torch.no_grad():
                # 计算预测分布 - 动态获取类别数
                _, predictions = torch.max(logits, dim=1)
                # 从logits维度或targets最大值获取类别数
                num_classes = max(logits.shape[1] if len(logits.shape) > 1 else 10, 
                                int(targets.max().item()) + 1 if len(targets) > 0 else 10)
                pred_counts = torch.bincount(predictions, minlength=num_classes)
                pred_distribution = pred_counts.float() / pred_counts.sum()
                
                # 计算熵（多样性指标）
                entropy = -torch.sum(pred_distribution * torch.log(pred_distribution + 1e-8)).item()
                max_entropy = np.log(num_classes)  # 使用动态类别数计算最大熵
                normalized_entropy = entropy / max_entropy
                
                # 检测崩溃模式
                max_class_ratio = pred_distribution.max().item()
                num_active_classes = (pred_distribution > 0.01).sum().item()  # 活跃类别数
                
                # 崩溃判断标准
                is_collapsed = (max_class_ratio > 0.8) or (num_active_classes < 3) or (normalized_entropy < 0.3)
                
                collapse_analysis = {
                    'round': round_idx,
                    'client_id': client_id,
                    'path_type': path_type,
                    'max_class_ratio': max_class_ratio,
                    'num_active_classes': num_active_classes,
                    'normalized_entropy': normalized_entropy,
                    'is_collapsed': is_collapsed,
                    'pred_distribution': pred_distribution.cpu().numpy().tolist()
                }
                
                self.metrics['classifier_collapse'].append(collapse_analysis)
                
                if is_collapsed:
                    self.collapse_detected = True
                    self.logger.error(f"轮次{round_idx} 客户端{client_id} {path_type}路径: 检测到分类器崩溃!")
                    self.logger.error(f"  - 最大类别占比: {max_class_ratio:.4f}")
                    self.logger.error(f"  - 活跃类别数: {num_active_classes}")
                    self.logger.error(f"  - 标准化熵: {normalized_entropy:.4f}")
                
                return collapse_analysis
                
        except Exception as e:
            self.logger.error(f"分类器崩溃检测失败: {str(e)}")
            return {}
    
    def analyze_classifier_weights(self, classifier_model, round_idx):
        """分析分类器权重分布"""
        try:
            weight_analysis = {
                'round': round_idx,
                'layers': {}
            }
            
            for name, param in classifier_model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    weight_data = param.data.cpu().float()
                    
                    # 基本统计
                    mean_val = weight_data.mean().item()
                    std_val = weight_data.std().item()
                    min_val = weight_data.min().item()
                    max_val = weight_data.max().item()
                    
                    # 权重分布分析
                    weight_range = max_val - min_val
                    zero_ratio = (weight_data.abs() < 1e-6).float().mean().item()
                    
                    # 检查权重崩溃
                    is_collapsed = (std_val < 1e-4) or (zero_ratio > 0.9) or (weight_range < 1e-4)
                    
                    layer_analysis = {
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'range': weight_range,
                        'zero_ratio': zero_ratio,
                        'is_collapsed': is_collapsed
                    }
                    
                    weight_analysis['layers'][name] = layer_analysis
                    
                    if is_collapsed:
                        self.logger.warning(f"轮次{round_idx} 分类器层{name}: 权重崩溃!")
                        self.logger.warning(f"  - 标准差: {std_val:.6f}")
                        self.logger.warning(f"  - 零值比例: {zero_ratio:.4f}")
            
            self.metrics['weight_distribution'].append(weight_analysis)
            return weight_analysis
            
        except Exception as e:
            self.logger.error(f"分类器权重分析失败: {str(e)}")
            return {}
    
    def analyze_shared_layer_quality(self, shared_features, round_idx, client_id):
        """分析共享层特征质量"""
        try:
            if shared_features is None or not isinstance(shared_features, torch.Tensor):
                return {}
            
            # 确保特征是2D
            if shared_features.dim() > 2:
                shared_features = F.adaptive_avg_pool2d(shared_features, (1, 1)).flatten(1)
            
            with torch.no_grad():
                # 特征激活统计
                mean_activation = shared_features.mean().item()
                std_activation = shared_features.std().item()
                max_activation = shared_features.max().item()
                min_activation = shared_features.min().item()
                
                # 死神经元检测
                dead_threshold = 1e-6
                dead_neurons = (shared_features.abs() < dead_threshold).all(dim=0).sum().item()
                total_neurons = shared_features.size(1)
                dead_ratio = dead_neurons / total_neurons if total_neurons > 0 else 0
                
                # 特征多样性
                feature_diversity = self._compute_feature_diversity_robust(shared_features)
                
                # 特征饱和度（激活值接近极值的比例）
                saturation_ratio = ((shared_features.abs() > 0.9 * max_activation).sum().item() / 
                                  shared_features.numel() if shared_features.numel() > 0 else 0)
                
                quality_analysis = {
                    'round': round_idx,
                    'client_id': client_id,
                    'mean_activation': mean_activation,
                    'std_activation': std_activation,
                    'activation_range': max_activation - min_activation,
                    'dead_neuron_ratio': dead_ratio,
                    'feature_diversity': feature_diversity,
                    'saturation_ratio': saturation_ratio,
                    'quality_score': 1.0 - dead_ratio - saturation_ratio
                }
                
                self.metrics['shared_layer_quality'].append(quality_analysis)
                
                # 检测质量问题
                if dead_ratio > 0.3:
                    self.logger.warning(f"轮次{round_idx} 客户端{client_id}: 共享层死神经元过多 ({dead_ratio:.2%})")
                
                if feature_diversity < 0.2:
                    self.logger.warning(f"轮次{round_idx} 客户端{client_id}: 共享层特征多样性不足 ({feature_diversity:.4f})")
                
                if saturation_ratio > 0.5:
                    self.logger.warning(f"轮次{round_idx} 客户端{client_id}: 共享层特征饱和度过高 ({saturation_ratio:.2%})")
                
                return quality_analysis
                
        except Exception as e:
            self.logger.error(f"共享层质量分析失败: {str(e)}")
            return {}
    
    def analyze_aggregation_weights(self, aggregation_weights, round_idx):
        """分析聚合权重分布"""
        try:
            if not aggregation_weights:
                return {}
            
            weights_array = np.array(list(aggregation_weights.values()))
            
            analysis = {
                'round': round_idx,
                'weights': dict(aggregation_weights),
                'mean_weight': weights_array.mean(),
                'std_weight': weights_array.std(),
                'max_weight': weights_array.max(),
                'min_weight': weights_array.min(),
                'weight_imbalance': weights_array.max() / (weights_array.min() + 1e-8),
                'is_balanced': weights_array.std() < 0.2
            }
            
            self.metrics['aggregation_analysis'].append(analysis)
            
            if analysis['weight_imbalance'] > 10:
                self.logger.warning(f"轮次{round_idx}: 聚合权重严重不平衡 (比例: {analysis['weight_imbalance']:.2f})")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"聚合权重分析失败: {str(e)}")
            return {}
    
    def monitor_learning_rates(self, client_lr_dict, round_idx):
        """监控学习率变化"""
        try:
            lr_analysis = {
                'round': round_idx,
                'learning_rates': dict(client_lr_dict),
                'avg_lr': np.mean(list(client_lr_dict.values())),
                'lr_std': np.std(list(client_lr_dict.values())),
                'max_lr': max(client_lr_dict.values()),
                'min_lr': min(client_lr_dict.values())
            }
            
            # 检查学习率是否过高或过低
            if lr_analysis['max_lr'] > 0.1:
                self.logger.warning(f"轮次{round_idx}: 学习率过高 (最大: {lr_analysis['max_lr']:.6f})")
            
            if lr_analysis['min_lr'] < 1e-6:
                self.logger.warning(f"轮次{round_idx}: 学习率过低 (最小: {lr_analysis['min_lr']:.6f})")
            
            return lr_analysis
            
        except Exception as e:
            self.logger.error(f"学习率监控失败: {str(e)}")
            return {}
    
    def comprehensive_diagnostic_report(self, round_idx):
        """生成综合诊断报告"""
        try:
            report = {
                'round': round_idx,
                'critical_issues': [],
                'warnings': [],
                'recommendations': [],
                'overall_health': 'unknown'
            }
            
            # 检查分类器崩溃
            recent_collapses = [m for m in self.metrics['classifier_collapse'] 
                              if m['round'] == round_idx and m['is_collapsed']]
            
            if recent_collapses:
                report['critical_issues'].append(f"检测到{len(recent_collapses)}个分类器崩溃")
                report['recommendations'].append("立即检查分类器架构和学习率设置")
            
            # 检查权重崩溃
            recent_weights = [m for m in self.metrics['weight_distribution'] if m['round'] == round_idx]
            collapsed_layers = 0
            for weight_analysis in recent_weights:
                for layer_name, layer_info in weight_analysis['layers'].items():
                    if layer_info['is_collapsed']:
                        collapsed_layers += 1
            
            if collapsed_layers > 0:
                report['critical_issues'].append(f"检测到{collapsed_layers}个权重崩溃层")
                report['recommendations'].append("降低学习率或增加权重初始化方差")
            
            # 检查特征质量
            recent_quality = [m for m in self.metrics['shared_layer_quality'] if m['round'] == round_idx]
            poor_quality_clients = [m for m in recent_quality if m['quality_score'] < 0.3]
            
            if poor_quality_clients:
                report['warnings'].append(f"{len(poor_quality_clients)}个客户端共享层质量较差")
                report['recommendations'].append("考虑调整共享层架构或增加正则化")
            
            # 检查梯度冲突
            recent_conflicts = [m for m in self.metrics['gradient_conflict'] 
                              if m['round'] == round_idx and m['avg_cosine_similarity'] < -0.1]
            
            if recent_conflicts:
                report['warnings'].append(f"检测到{len(recent_conflicts)}个严重梯度冲突")
                report['recommendations'].append("调整损失权重平衡或使用梯度裁剪")
            
            # 整体健康评估
            if report['critical_issues']:
                report['overall_health'] = 'critical'
            elif len(report['warnings']) > 2:
                report['overall_health'] = 'poor'
            elif report['warnings']:
                report['overall_health'] = 'warning'
            else:
                report['overall_health'] = 'good'
            
            return report
            
        except Exception as e:
            self.logger.error(f"综合诊断报告生成失败: {str(e)}")
            return {'round': round_idx, 'overall_health': 'error'}
    
    def export_metrics_to_wandb(self, wandb_logger):
        """导出核心指标到wandb"""
        try:
            if not self.metrics['classifier_collapse']:
                return
            
            latest_round = max([m['round'] for m in self.metrics['classifier_collapse']])
            
            # 分类器崩溃指标
            recent_collapses = [m for m in self.metrics['classifier_collapse'] if m['round'] == latest_round]
            if recent_collapses:
                collapse_count = sum([1 for m in recent_collapses if m['is_collapsed']])
                avg_entropy = np.mean([m['normalized_entropy'] for m in recent_collapses])
                avg_active_classes = np.mean([m['num_active_classes'] for m in recent_collapses])
                
                wandb_logger.log({
                    "diagnostic/classifier_collapses": collapse_count,
                    "diagnostic/avg_prediction_entropy": avg_entropy,
                    "diagnostic/avg_active_classes": avg_active_classes,
                    "round": latest_round
                })
            
            # 特征质量指标
            recent_quality = [m for m in self.metrics['shared_layer_quality'] if m['round'] == latest_round]
            if recent_quality:
                avg_quality = np.mean([m['quality_score'] for m in recent_quality])
                avg_dead_ratio = np.mean([m['dead_neuron_ratio'] for m in recent_quality])
                
                wandb_logger.log({
                    "diagnostic/avg_feature_quality": avg_quality,
                    "diagnostic/avg_dead_neuron_ratio": avg_dead_ratio,
                    "round": latest_round
                })
            
            # 权重分布指标
            recent_weights = [m for m in self.metrics['weight_distribution'] if m['round'] == latest_round]
            if recent_weights:
                collapsed_layers = sum([sum([1 for layer_info in m['layers'].values() if layer_info['is_collapsed']]) 
                                      for m in recent_weights])
                wandb_logger.log({
                    "diagnostic/collapsed_weight_layers": collapsed_layers,
                    "round": latest_round
                })
                
        except Exception as e:
            self.logger.error(f"导出wandb指标失败: {str(e)}")
    
    # 辅助方法
    def _classify_conflict_level(self, cosine_similarity):
        """分类冲突级别"""
        if cosine_similarity > 0.5:
            return "协同"
        elif cosine_similarity > 0:
            return "轻微冲突"
        elif cosine_similarity > -0.5:
            return "中等冲突"
        else:
            return "严重冲突"
    
    def _compute_feature_diversity_robust(self, features):
        """稳健的特征多样性计算"""
        try:
            if features.size(0) < 2:
                return 0.5
            
            # 计算特征间的相关性
            normalized_features = F.normalize(features, dim=1)
            correlation_matrix = torch.mm(normalized_features, normalized_features.t())
            
            # 排除对角线元素
            mask = torch.eye(correlation_matrix.size(0), device=correlation_matrix.device).bool()
            off_diagonal = correlation_matrix[~mask]
            
            # 计算多样性（1 - 平均相关性）
            avg_correlation = off_diagonal.mean().item()
            diversity = 1.0 - abs(avg_correlation)
            
            return max(0.0, min(1.0, diversity))
            
        except Exception as e:
            return 0.5
    
    def _compute_stability_score(self, avg_change, avg_norm):
        """计算稳定性得分"""
        try:
            if avg_norm < 1e-8:
                return 0.0
            relative_change = avg_change / avg_norm
            stability_score = 1.0 / (1.0 + 10 * relative_change)
            return max(0.0, min(1.0, stability_score))
        except:
            return 0.5

@torch.no_grad()
def compute_client_feature_similarity_robust(client_models, server_feature_extractor, sample_loader, device):
    """修复版客户端特征相似度计算，确保每个客户端使用自己的本地特征提取路径"""
    # 取一个小批次做对比
    try:
        images, _ = next(iter(sample_loader))
        images = images.to(device)
    except:
        return 0.5  # 返回中性值如果无法获取数据

    feats = []
    for cid in sorted(client_models.keys()):
        try:
            cm = client_models[cid].to(device)
            cm.eval()
            
            # 🔥 关键修复：使用客户端本地的共享/个性化特征，而不是统一的服务器特征
            local_logits, shared_features, personal_features = cm(images)
            
            # 使用客户端共享层特征作为客户端特色表征
            f_client = shared_features
            
            # 可选：如果有服务器特征提取器，再接上
            if server_feature_extractor is not None:
                f = server_feature_extractor(f_client)
            else:
                f = f_client
                
            # 展平 + L2 归一化，避免尺度影响
            f = f.view(f.size(0), -1)
            f = torch.nn.functional.normalize(f, dim=1)
            
            # 聚成一个 batch 的均值特征，用于客户端级别表示
            feats.append(f.mean(dim=0))  # [D]
            
        except Exception as e:
            # 如果某个客户端出错，跳过
            continue
    
    if len(feats) < 2:
        return 0.5  # 需要至少两个客户端才能计算相似度
    
    # 计算客户端间平均余弦相似度
    F = torch.stack(feats, dim=0)  # [K, D]
    sim = torch.mm(F, F.t()).clamp(-1, 1)  # 余弦：向量已归一化
    
    # 提取上三角矩阵（除去对角线）
    upper_indices = torch.triu(torch.ones_like(sim), diagonal=1).bool()
    upper_sim = sim[upper_indices]
    
    if len(upper_sim) == 0:
        return 0.5
        
    avg_similarity = float(upper_sim.mean().item())
    return avg_similarity