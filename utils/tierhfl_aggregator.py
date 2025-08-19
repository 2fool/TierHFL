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

    def aggregate_shared_layers(self, shared_states: dict, eval_results: dict):
        client_weights = {cid: float(res.get("global_accuracy", 0.0)) for cid, res in eval_results.items()}
        s = sum(client_weights.values())
        if s <= 0:
            n = len(eval_results) or 1
            client_weights = {cid: 1.0 / n for cid in eval_results.keys()}
        else:
            client_weights = {cid: w / s for cid, w in client_weights.items()}
        return self._weighted_average(shared_states, client_weights)

    def aggregate_server_models(self, cluster_server_models: dict, eval_results: dict=None, beta: float=0.9):
        cluster_weights = {}
        if eval_results:
            for cid, res in eval_results.items():
                cluster_id = res.get("cluster_id", 0)
                acc = float(res.get("global_accuracy", 0.0))
                cluster_weights[cluster_id] = cluster_weights.get(cluster_id, 0.0) + max(acc, 0.0)
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
            for cid, res in eval_results.items():
                cluster_id = res.get("cluster_id", 0)
                acc = float(res.get("global_accuracy", 0.0))
                cluster_weights[cluster_id] = cluster_weights.get(cluster_id, 0.0) + max(acc, 0.0)
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
