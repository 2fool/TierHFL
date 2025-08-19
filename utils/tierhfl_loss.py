import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# =========================
# 1) 标签平滑交叉熵（用于全局路径）
# =========================
def cross_entropy_ls(logits, target, eps=0.1):
    """
    标签平滑版交叉熵，缓解非IID/长尾导致的过拟合与塌缩。
    - logits: [B, C]
    - target: [B]
    """
    n_class = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logp)
        true_dist.fill_(eps / max(1, (n_class - 1)))
        true_dist.scatter_(1, target.unsqueeze(1), 1 - eps)
    return -(true_dist * logp).sum(dim=1).mean()

# =========================
# 2) 梯度投影器（原样保留）
# =========================
class GradientProjector:
    """梯度投影器，处理共享层的梯度冲突"""
    def __init__(self, similarity_threshold=0.3, projection_frequency=5):
        self.similarity_threshold = similarity_threshold
        self.projection_frequency = projection_frequency
        self.batch_count = 0

    def should_project(self):
        """判断是否需要进行梯度投影"""
        self.batch_count += 1
        return self.batch_count % self.projection_frequency == 0

    def compute_cosine_similarity(self, grad1, grad2):
        """计算两个梯度的余弦相似度"""
        if grad1 is None or grad2 is None:
            return 1.0
        g1_flat = grad1.flatten()
        g2_flat = grad2.flatten()
        cos_sim = F.cosine_similarity(g1_flat.unsqueeze(0), g2_flat.unsqueeze(0))
        return cos_sim.item()

    def project_gradient(self, g_personal, g_global, alpha=0.5):
        """梯度投影：将个性化梯度投影到全局梯度方向"""
        if g_personal is None or g_global is None:
            return g_global if g_global is not None else g_personal
        g_p_flat = g_personal.flatten()
        g_g_flat = g_global.flatten()
        dot_product = torch.dot(g_p_flat, g_g_flat)
        g_g_norm_sq = torch.dot(g_g_flat, g_g_flat)
        if g_g_norm_sq > 1e-8:
            projection = (dot_product / g_g_norm_sq) * g_g_flat
            projected_g_p = alpha * projection + (1 - alpha) * g_p_flat
            return projected_g_p.view_as(g_personal)
        else:
            return g_personal

    def process_shared_gradients(self, model, personal_loss, global_loss, alpha_stage=0.5):
        """处理共享层梯度冲突"""
        if not self.should_project():
            return False

        # 获取共享层参数
        shared_params = []
        for name, param in model.named_parameters():
            if 'shared_base' in name and param.requires_grad:
                shared_params.append((name, param))
        if not shared_params:
            return False

        # 计算个性化路径梯度
        personal_grads = torch.autograd.grad(
            personal_loss, [param for _, param in shared_params],
            retain_graph=True, allow_unused=True
        )
        # 计算全局路径梯度
        global_grads = torch.autograd.grad(
            global_loss, [param for _, param in shared_params],
            retain_graph=True, allow_unused=True
        )

        # 处理梯度冲突
        conflicts_resolved = 0
        for i, ((name, param), g_p, g_g) in enumerate(zip(shared_params, personal_grads, global_grads)):
            if g_p is not None and g_g is not None:
                cos_sim = self.compute_cosine_similarity(g_p, g_g)
                if cos_sim < self.similarity_threshold:
                    projected_grad = self.project_gradient(g_p, g_g, alpha_stage)
                    param.grad = projected_grad
                    conflicts_resolved += 1
                else:
                    param.grad = alpha_stage * g_p + (1 - alpha_stage) * g_g

        if conflicts_resolved > 0:
            logging.debug(f"解决了 {conflicts_resolved} 个梯度冲突")

        return True

# =========================
# 3) 特征平衡损失（原样保留）
# =========================
class FeatureBalanceLoss(nn.Module):
    """特征平衡损失，确保共享层对两条路径都有用"""
    def __init__(self, temperature=1.0):
        super(FeatureBalanceLoss, self).__init__()
        self.temperature = temperature

    def compute_feature_importance(self, features, gradients):
        """计算特征重要性（使用梯度模长作为代理）"""
        if gradients is None:
            return torch.zeros(1, device=features.device)
        grad_norm = torch.norm(gradients.flatten())
        return grad_norm

    def forward(self, shared_features, personal_gradients, global_gradients):
        """计算特征平衡损失"""
        personal_importance = self.compute_feature_importance(shared_features, personal_gradients)
        global_importance = self.compute_feature_importance(shared_features, global_gradients)
        total_importance = personal_importance + global_importance + 1e-8
        personal_ratio = personal_importance / total_importance
        global_ratio = global_importance / total_importance
        balance_loss = torch.abs(personal_ratio - 0.5) + torch.abs(global_ratio - 0.5)
        return balance_loss

# =========================
# 4) 分阶段损失（改动点：全局CE→标签平滑 + 熵正则）
# =========================
class EnhancedStagedLoss(nn.Module):
    """
    增强版分阶段损失函数
    - 改动：全局路径使用标签平滑 + 熵正则（防塌缩）
    - 接口保持不变
    """
    def __init__(self, ls_eps: float = 0.1, entropy_coeff: float = 1e-3):
        super(EnhancedStagedLoss, self).__init__()
        self.ce_loss_local = nn.CrossEntropyLoss()   # 本地路径仍用标准CE
        self.feature_balance_loss = FeatureBalanceLoss()
        self.gradient_projector = GradientProjector()
        self.lambda_balance = 0.1

        # 反塌缩参数
        self.ls_eps = ls_eps
        self.entropy_coeff = entropy_coeff

    # ---------- 阶段1：纯全局特征学习 ----------
    def stage1_loss(self, global_logits, targets, shared_features=None):
        """
        返回: total_loss, global_loss, feature_importance_loss
        """
        # 全局路径：标签平滑
        global_loss = cross_entropy_ls(global_logits, targets, eps=self.ls_eps)

        # 特征多样性（避免早期特征坍塌）
        feature_importance_loss = torch.tensor(0.0, device=global_logits.device)
        if shared_features is not None:
            # 单样本内的通道/空间方差越大越好：用 -log(std) 作为惩罚
            features_flat = shared_features.flatten(1)
            feature_std = torch.std(features_flat, dim=1).mean()
            feature_importance_loss = -torch.log(feature_std + 1e-8)

        # 熵正则（越大越好，取负号加入总loss）
        with torch.no_grad():
            pass  # 仅注释对齐
        probs = torch.softmax(global_logits, dim=-1).clamp_min(1e-6)
        entropy = -(probs * probs.log()).sum(dim=1).mean()

        total_loss = global_loss - self.entropy_coeff * entropy + 0.05 * feature_importance_loss
        return total_loss, global_loss, feature_importance_loss

    # ---------- 阶段2&3：交替训练/精细调整 ----------
    def stage2_3_loss(self, local_logits, global_logits, targets,
                      personal_gradients=None, global_gradients=None,
                      shared_features=None, alpha=0.5):
        """
        返回: total_loss, local_loss, global_loss, balance_loss
        """
        # 本地头：保持标准 CE（个性化更贴近本地分布）
        local_loss = self.ce_loss_local(local_logits, targets)
        # 全局头：标签平滑 + 熵正则
        global_ce = cross_entropy_ls(global_logits, targets, eps=self.ls_eps)
        probs = torch.softmax(global_logits, dim=-1).clamp_min(1e-6)
        entropy = -(probs * probs.log()).sum(dim=1).mean()
        global_loss = global_ce - self.entropy_coeff * entropy

        # 特征平衡损失（如果提供了两路梯度）
        balance_loss = torch.tensor(0.0, device=global_logits.device)
        if shared_features is not None and personal_gradients is not None and global_gradients is not None:
            balance_loss = self.feature_balance_loss(shared_features, personal_gradients, global_gradients)

        total_loss = alpha * local_loss + (1 - alpha) * global_loss + self.lambda_balance * balance_loss
        return total_loss, local_loss, global_loss, balance_loss

    # ---------- 梯度投影（原样保留） ----------
    def apply_gradient_projection(self, model, personal_loss, global_loss, alpha_stage=0.5):
        """应用梯度投影"""
        return self.gradient_projector.process_shared_gradients(
            model, personal_loss, global_loss, alpha_stage
        )
