import torch

class LayeredAggregator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.server_momentum = {}
        self.classifier_momentum = {}

    def _is_bn_buffer(self, key: str) -> bool:
        return any(tag in key for tag in ["running_mean", "running_var", "num_batches_tracked"])

    def _weighted_average(self, state_dicts: dict, weights: dict):
        if not state_dicts:
            return {}
        common_keys = None
        for sd in state_dicts.values():
            kset = set(sd.keys())
            common_keys = kset if common_keys is None else (common_keys & kset)
        if not common_keys:
            return {}
        out = {}
        for key in common_keys:
            if self._is_bn_buffer(key):
                continue
            acc, wsum, acc_dtype = None, 0.0, None
            for sid, sd in state_dicts.items():
                if key not in sd: continue
                t = sd[key]
                if not torch.is_tensor(t) or not t.dtype.is_floating_point:
                    continue
                w = float(weights.get(sid, 0.0))
                if w <= 0.0: continue
                t32 = t.detach().to(self.device, dtype=torch.float32)
                acc = (w * t32) if acc is None else (acc + w * t32)
                wsum += w
                acc_dtype = t.dtype
            if acc is not None and wsum > 0:
                out[key] = (acc / wsum).to(dtype=acc_dtype)
        return out

    def aggregate_shared_layers(self, shared_states: dict, eval_results: dict=None):
        """聚合客户端共享层，加入None兜底逻辑"""
        if not shared_states:
            return {}
            
        # 当eval_results为空时，使用均匀权重兜底（与server/classifier的兜底一致）
        if not eval_results:
            n = len(shared_states) or 1
            client_weights = {cid: 1.0 / n for cid in shared_states.keys()}
        else:
            # 与现有实现保持一致：样本数 * 精度 + epsilon
            client_weights = self._build_weights(
                eval_results,
                key_samples="train_samples",
                key_acc="global_accuracy",
                use_acc_weight=True,
                eps=1e-9
            )
        
        return self._weighted_average(shared_states, client_weights)

    def aggregate_server_models(self, cluster_server_models: dict, eval_results: dict=None, beta: float=0.9):
        cluster_weights = {}
        if eval_results:
            # 按聚类汇总权重
            cluster_samples = {}
            cluster_accuracy = {}
            for cid, res in eval_results.items():
                cluster_id = res.get("cluster_id", 0)
                samples = max(float(res.get("train_samples", 1)), 1)  # 避免除零
                acc = max(float(res.get("global_accuracy", 0.0)), 0.0)
                cluster_samples[cluster_id] = cluster_samples.get(cluster_id, 0) + samples
                cluster_accuracy[cluster_id] = cluster_accuracy.get(cluster_id, 0) + acc * samples
            
            for cluster_id in cluster_samples:
                avg_acc = cluster_accuracy[cluster_id] / max(cluster_samples[cluster_id], 1)
                cluster_weights[cluster_id] = cluster_samples[cluster_id] * (avg_acc + 0.01)  # 加epsilon避免零权重
        
        s = sum(cluster_weights.values())
        if s <= 0:
            n = len(cluster_server_models) or 1
            cluster_weights = {k: 1.0 / n for k in cluster_server_models.keys()}
        else:
            cluster_weights = {k: v / s for k, v in cluster_weights.items()}
        
        avg_state = self._weighted_average(cluster_server_models, cluster_weights)
        if not self.server_momentum:
            self.server_momentum = {k: v.clone() for k, v in avg_state.items()}
        else:
            for k, v in avg_state.items():
                self.server_momentum[k] = beta * self.server_momentum.get(k, v) + (1 - beta) * v
        return self.server_momentum

    def aggregate_global_classifiers(self, cluster_classifiers: dict, eval_results: dict=None, beta: float=0.9):
        cluster_weights = {}
        if eval_results:
            # 按聚类汇总权重
            cluster_samples = {}
            cluster_accuracy = {}
            for cid, res in eval_results.items():
                cluster_id = res.get("cluster_id", 0)
                samples = max(float(res.get("train_samples", 1)), 1)  # 避免除零
                acc = max(float(res.get("global_accuracy", 0.0)), 0.0)
                cluster_samples[cluster_id] = cluster_samples.get(cluster_id, 0) + samples
                cluster_accuracy[cluster_id] = cluster_accuracy.get(cluster_id, 0) + acc * samples
            
            for cluster_id in cluster_samples:
                avg_acc = cluster_accuracy[cluster_id] / max(cluster_samples[cluster_id], 1)
                cluster_weights[cluster_id] = cluster_samples[cluster_id] * (avg_acc + 0.01)  # 加epsilon避免零权重
        
        s = sum(cluster_weights.values())
        if s <= 0:
            n = len(cluster_classifiers) or 1
            cluster_weights = {k: 1.0 / n for k in cluster_classifiers.keys()}
        else:
            cluster_weights = {k: v / s for k, v in cluster_weights.items()}
        
        avg_state = self._weighted_average(cluster_classifiers, cluster_weights)
        if not self.classifier_momentum:
            self.classifier_momentum = {k: v.clone() for k, v in avg_state.items()}
        else:
            for k, v in avg_state.items():
                self.classifier_momentum[k] = beta * self.classifier_momentum.get(k, v) + (1 - beta) * v
        return self.classifier_momentum

    def _build_weights(self, eval_results: dict, key_samples="train_samples", key_acc="global_accuracy", use_acc_weight=True, eps=1e-9):
        """构建聚合权重，综合考虑样本数量和准确率"""
        weights = {}
        for cid, res in eval_results.items():
            samples = max(float(res.get(key_samples, 1)), 1)  # 样本数，避免除零
            w = samples
            if use_acc_weight:
                acc = max(float(res.get(key_acc, 0.0)), 0.0)
                w *= (acc + eps)  # 精度权重，加epsilon避免零权重
            weights[cid] = w
        
        # 归一化
        s = sum(weights.values()) + eps
        return {cid: w / s for cid, w in weights.items()}
