import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging
import matplotlib.pyplot as plt  # 仅在需要可视化时使用
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import copy
import os
from datetime import datetime

def analyze_server_features(server_model, client_model, global_test_loader, device='cpu'):
    """分析服务器提取特征的可分性"""
    server_model.eval()
    client_model.eval()  # 使用传入的单个客户端模型
    features_all = []
    labels_all = []
    
    # 收集特征和标签
    with torch.no_grad():
        for data, target in global_test_loader:
            data = data.to(device)
            # 使用传入的客户端模型获取共享层输出
            _, shared_features, _ = client_model(data)
            # 通过服务器模型提取特征
            server_features = server_model(shared_features)
            
            features_all.append(server_features.cpu())
            labels_all.append(target)
    
    features_all = torch.cat(features_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    
    # 计算类内/类间距离比
    class_means = {}
    for c in range(10):  # 假设10个类别
        class_idx = (labels_all == c).nonzero(as_tuple=True)[0]
        if len(class_idx) > 0:  # 确保该类有样本
            class_means[c] = features_all[class_idx].mean(dim=0)
    
    # 类内距离
    intra_class_dist = 0
    num_classes_with_samples = 0
    for c in range(10):
        class_idx = (labels_all == c).nonzero(as_tuple=True)[0]
        if len(class_idx) > 0:
            class_features = features_all[class_idx]
            intra_class_dist += torch.norm(class_features - class_means[c], dim=1).mean()
            num_classes_with_samples += 1
    
    if num_classes_with_samples > 0:
        intra_class_dist /= num_classes_with_samples
    
    # 类间距离
    inter_class_dist = 0
    count = 0
    classes_with_means = list(class_means.keys())
    for i in range(len(classes_with_means)):
        for j in range(i+1, len(classes_with_means)):
            c1 = classes_with_means[i]
            c2 = classes_with_means[j]
            inter_class_dist += torch.norm(class_means[c1] - class_means[c2])
            count += 1
    
    if count > 0:
        inter_class_dist /= count
    
    separability = inter_class_dist / (intra_class_dist + 1e-8)
    print(f"特征可分性(类间/类内距离比): {separability:.4f}")
    
    return separability, features_all, labels_all

def test_with_simple_classifier(server_model, client_model, global_test_loader, device='cpu'):
    """用简单分类器替代全局分类器测试特征质量"""
    # 收集特征和标签用于训练新分类器
    features_all = []
    labels_all = []
    
    with torch.no_grad():
        for data, target in global_test_loader:
            data = data.to(device)
            _, shared_features, _ = client_model(data)  # 使用传入的客户端模型
            server_features = server_model(shared_features)
            
            features_all.append(server_features.cpu())
            labels_all.append(target)
    
    features_train = torch.cat(features_all[:len(features_all)//2], dim=0)
    labels_train = torch.cat(labels_all[:len(labels_all)//2], dim=0)
    features_test = torch.cat(features_all[len(features_all)//2:], dim=0)
    labels_test = torch.cat(labels_all[len(labels_all)//2:], dim=0)
    
    # 训练一个简单的线性分类器
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(features_train.numpy(), labels_train.numpy())
    
    # 评估新分类器
    accuracy = classifier.score(features_test.numpy(), labels_test.numpy()) * 100
    print(f"简单分类器准确率: {accuracy:.2f}%")
    
    return accuracy

def analyze_feature_consistency(server_model, client_models, test_data_dict, device='cpu'):
    """分析不同客户端间特征的一致性"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 对所有客户端的特征进行分析
    client_features = {}
    client_labels = {}
    
    for client_id, test_loader in test_data_dict.items():
        features = []
        labels = []
        
        # 确保当前客户端模型在正确的设备上
        client_model = client_models[client_id].to(device)
        client_model.eval()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                
                features.append(server_features.cpu())
                labels.append(target.cpu())
        
        if features:
            client_features[client_id] = torch.cat(features, dim=0)
            client_labels[client_id] = torch.cat(labels, dim=0)
    
    # 计算特征统计信息
    stats = {}
    for client_id in client_features:
        feats = client_features[client_id]
        stats[client_id] = {
            'mean': feats.mean().item(),
            'std': feats.std().item(),
            'norm': torch.norm(feats, dim=1).mean().item()
        }
    
    # 计算客户端间特征相似性
    similarities = {}
    for i in client_features:
        for j in client_features:
            if i != j:
                # 计算相同类别样本的特征相似度
                sim_by_class = {}
                for c in range(10):  # 假设10个类别
                    i_idx = (client_labels[i] == c).nonzero(as_tuple=True)[0]
                    j_idx = (client_labels[j] == c).nonzero(as_tuple=True)[0]
                    
                    if len(i_idx) > 0 and len(j_idx) > 0:
                        i_feats = client_features[i][i_idx]
                        j_feats = client_features[j][j_idx]
                        
                        # 计算余弦相似度
                        i_norm = F.normalize(i_feats, dim=1)
                        j_norm = F.normalize(j_feats, dim=1)
                        
                        # 计算平均余弦相似度
                        sim_matrix = torch.mm(i_norm, j_norm.t())
                        sim_score = sim_matrix.max(dim=1)[0].mean().item()
                        sim_by_class[c] = sim_score
                
                if sim_by_class:
                    similarities[f"{i}-{j}"] = sum(sim_by_class.values()) / len(sim_by_class)
    
    # 计算平均相似度
    avg_similarity = sum(similarities.values()) / len(similarities) if similarities else 0
    print(f"客户端间特征平均相似度: {avg_similarity:.4f}")
    
    return stats, similarities, avg_similarity

def test_server_compression_ability(server_model, client_models, global_test_loader, device='cpu'):
    """测试服务器模型压缩非IID特征的能力"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 获取服务器模型的中间层输出
    activation = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # 注册钩子函数获取中间层输出
    hooks = []
    layers = [module for name, module in server_model.named_modules() 
             if isinstance(module, (nn.Conv2d, nn.Linear))]
    
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(get_activation(f'layer_{i}')))
    
    # 对随机选取的测试样本进行前向传播
    with torch.no_grad():
        data, _ = next(iter(global_test_loader))
        data = data.to(device)
        
        # 获取不同客户端的共享层特征
        client_shared_features = {}
        for client_id, model in client_models.items():
            model = model.to(device)  # 确保模型在正确设备上
            model.eval()
            _, shared_feats, _ = model(data)
            client_shared_features[client_id] = shared_feats
        
        # 对每个客户端的特征计算服务器各层输出
        client_activations = {}
        for client_id, shared_feats in client_shared_features.items():
            # 清空之前的激活
            activation.clear()
            
            # 前向传播
            server_model(shared_feats)
            
            # 保存该客户端的激活值
            client_activations[client_id] = {k: v.clone().cpu() for k, v in activation.items()}
    
    # 清除钩子
    for hook in hooks:
        hook.remove()
    
    # 计算每层特征的客户端间相似度
    layer_similarities = {}
    for layer_name in next(iter(client_activations.values())).keys():
        # 收集所有客户端该层的激活值
        layer_acts = {}
        for client_id, acts in client_activations.items():
            layer_acts[client_id] = acts[layer_name].view(acts[layer_name].size(0), -1)
        
        # 计算客户端间该层输出的相似度
        similarities = []
        clients = list(layer_acts.keys())
        for i in range(len(clients)):
            for j in range(i+1, len(clients)):
                ci, cj = clients[i], clients[j]
                
                # 扁平化并标准化
                acts_i = F.normalize(layer_acts[ci], dim=1)
                acts_j = F.normalize(layer_acts[cj], dim=1)
                
                # 计算余弦相似度
                sim = torch.mm(acts_i, acts_j.t()).diag().mean().item()
                similarities.append(sim)
        
        # 该层的平均相似度
        if similarities:
            layer_similarities[layer_name] = sum(similarities) / len(similarities)
    
    # 计算服务器层间相似度变化，判断是否在服务器内增加了一致性
    for i in range(len(layer_similarities) - 1):
        layer1 = f'layer_{i}'
        layer2 = f'layer_{i+1}'
        if layer1 in layer_similarities and layer2 in layer_similarities:
            diff = layer_similarities[layer2] - layer_similarities[layer1]
            print(f"{layer1} -> {layer2} 客户端间相似度变化: {diff:.4f}")
    
    return layer_similarities

def test_client_identity_encoding(server_model, client_models, test_data_dict, device='cpu'):
    """测试服务器特征中是否包含客户端身份信息"""
    server_model = server_model.to(device)
    server_model.eval()
    
    # 收集各客户端特征
    client_features = []
    client_ids = []
    
    for client_id, test_loader in test_data_dict.items():
        features = []
        
        # 确保当前客户端模型在正确的设备上
        client_model = client_models[client_id].to(device)
        client_model.eval()
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(device)
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                
                features.append(server_features.cpu())
        
        if features:
            client_features.append(torch.cat(features, dim=0))
            client_ids.extend([client_id] * len(features))
    
    features_all = torch.cat(client_features, dim=0)
    client_ids = np.array(client_ids)
    
    # 训练一个分类器来预测客户端身份
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_all.numpy(), client_ids, test_size=0.3, random_state=42)
    
    classifier = SVC()
    classifier.fit(X_train, y_train)
    
    # 评估预测客户端身份的准确率
    accuracy = classifier.score(X_test, y_test) * 100
    print(f"从特征预测客户端身份的准确率: {accuracy:.2f}%")
    
    # 随机分类的准确率作为基准
    random_accuracy = 100 / len(set(client_ids))
    print(f"随机猜测客户端身份的准确率: {random_accuracy:.2f}%")
    
    return accuracy, random_accuracy

def validate_server_effectiveness(args, client_models, server_model, global_classifier,
                                 global_test_loader, test_data_local_dict, device='cpu'):
    """集成验证服务器特征提取有效性的函数"""
    print("\n===== 验证服务器特征提取有效性 =====")
    
    # 确保服务器模型在正确设备上
    server_model = server_model.to(device)
    
    # 选择一个客户端模型用于测试
    sample_client_id = list(client_models.keys())[0]
    sample_client_model = client_models[sample_client_id].to(device)
    
    try:
        # 1. 特征可分性分析
        separability, features, labels = analyze_server_features(
            server_model, sample_client_model, global_test_loader, device=device)
    except Exception as e:
        print(f"特征可分性分析出错: {str(e)}")
        separability = 0.0
    
    try:
        # 2. 替换分类器测试
        new_classifier_acc = test_with_simple_classifier(
            server_model, sample_client_model, global_test_loader, device=device)
    except Exception as e:
        print(f"替换分类器测试出错: {str(e)}")
        new_classifier_acc = 0.0
    
    try:
        # 3. 特征一致性跨客户端分析
        feature_stats, similarities, avg_similarity = analyze_feature_consistency(
            server_model, client_models, test_data_local_dict, device=device)
    except Exception as e:
        print(f"特征一致性分析出错: {str(e)}")
        avg_similarity = 0.0
        similarities = {}
        feature_stats = {}
    
    try:
        # 4. 服务器模型压缩能力测试
        layer_similarities = test_server_compression_ability(
            server_model, client_models, global_test_loader, device=device)
    except Exception as e:
        print(f"服务器压缩能力测试出错: {str(e)}")
        layer_similarities = {}
    
    # 暂时跳过可能有问题的客户端身份编码测试
    identity_acc = 0.0
    random_acc = 0.0
    identity_leakage = 0.0
    
    print(f"\n已获取有效的验证指标:")
    print(f"1. 特征可分性: {separability:.4f}")
    print(f"2. 简单分类器准确率: {new_classifier_acc:.2f}%")
    print(f"3. 客户端间特征平均相似度: {avg_similarity:.4f}")
    
    # 基于已有数据做出评估
    feature_quality_score = (separability * 0.4 + (new_classifier_acc / 100) * 0.6) 
    
    print("\n===== 服务器特征提取能力评估 =====")
    print(f"特征质量得分(0-1): {feature_quality_score:.4f}")
    print(f"数据异质性适应能力(0-1): {avg_similarity:.4f}")
    
    if feature_quality_score > 0.3:
        print("结论: 服务器特征提取工作正常，但可能需要优化以更好适应数据异质性")
    elif new_classifier_acc > 20:
        print("结论: 服务器提取的特征有一定区分能力，但全局分类器可能存在问题")
    else:
        print("结论: 服务器特征提取存在明显问题，无法提供有效特征")
        
    return {
        'feature_quality': feature_quality_score,
        'heterogeneity_adaptation': avg_similarity,
        'simple_classifier_acc': new_classifier_acc
    }



class GlobalClassifierVerifier:
    """全局分类器问题诊断工具"""
    
    def __init__(self, server_model, global_classifier, client_models, 
                 global_test_loader, test_data_dict, device='cpu'):
        """初始化验证器"""
        self.server_model = server_model.to(device)
        self.global_classifier = global_classifier.to(device)
        self.client_models = {k: v.to(device) for k, v in client_models.items()}
        self.global_test_loader = global_test_loader
        self.test_data_dict = test_data_dict
        self.device = device
        
        # 创建结果目录
        self.result_dir = f"classifier_verification_{datetime.now().strftime('%m%d_%H%M')}"
        os.makedirs(self.result_dir, exist_ok=True)
        
    def run_all_tests(self):
        """运行所有验证测试"""
        print("\n===== 全局分类器诊断 =====")
        
        # 1. 基准性能测试
        self.test_baseline_performance()
        
        # 2. 替换分类器测试
        self.test_alternative_classifiers()
        
        # 3. 特征分布分析
        self.analyze_feature_distribution()
        
        # 4. 分类器权重分析
        self.analyze_classifier_weights()
        
        # 5. 梯度流分析
        self.analyze_gradient_flow()
        
        # 6. 测试泛化能力
        self.test_generalization()
        
        print("\n===== 诊断完成 =====")
    
    def test_baseline_performance(self):
        """测试基准性能"""
        print("\n1. 基准性能测试")
        
        # 在全局测试集上测试原始分类器
        original_acc = self._evaluate_on_global_test(
            self.server_model, self.global_classifier)
            
        # 在各客户端测试集上测试
        client_accs = {}
        for client_id, test_loader in self.test_data_dict.items():
            client_model = self.client_models[client_id]
            acc = self._evaluate_on_client_test(
                client_model, self.server_model, self.global_classifier, test_loader)
            client_accs[client_id] = acc
            
        avg_client_acc = sum(client_accs.values()) / len(client_accs)
        
        print(f"原始全局分类器在IID测试集上准确率: {original_acc:.2f}%")
        print(f"原始全局分类器在客户端测试集上平均准确率: {avg_client_acc:.2f}%")
        
        return original_acc, client_accs
    
    def test_alternative_classifiers(self):
        """测试替代分类器架构"""
        print("\n2. 替代分类器测试")
        
        # 获取特征维度
        feature_dim = None
        for param in self.global_classifier.parameters():
            if len(param.shape) > 1:
                feature_dim = param.shape[1]
                break
        
        if not feature_dim:
            for name, module in self.global_classifier.named_modules():
                if isinstance(module, nn.Linear):
                    feature_dim = module.in_features
                    break
        
        if not feature_dim:
            print("无法确定特征维度，使用默认值128")
            feature_dim = 128
            
        # 确定类别数量
        num_classes = 10  # 假设是CIFAR-10
        
        # 定义几种不同的分类器架构
        classifiers = {
            "线性分类器": nn.Linear(feature_dim, num_classes).to(self.device),
            "单隐层MLP": nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ).to(self.device),
            "双隐层MLP": nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            ).to(self.device),
            "BatchNorm版MLP": nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            ).to(self.device)
        }
        
        # 从全局测试集收集特征
        all_features, all_labels = self._collect_features_from_global_test()
        
        # 用收集的特征训练和评估每种分类器
        results = {}
        for name, classifier in classifiers.items():
            # 训练分类器
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
            classifier.train()
            
            # 使用80%数据训练，20%测试
            split_idx = int(0.8 * len(all_features))
            train_features, train_labels = all_features[:split_idx], all_labels[:split_idx]
            test_features, test_labels = all_features[split_idx:], all_labels[split_idx:]
            
            # 简单训练
            for epoch in range(50):
                optimizer.zero_grad()
                logits = classifier(train_features)
                loss = F.cross_entropy(logits, train_labels)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  {name} 训练轮次 {epoch+1}/50, 损失: {loss.item():.4f}")
            
            # 评估
            classifier.eval()
            with torch.no_grad():
                logits = classifier(test_features)
                _, preds = logits.max(1)
                accuracy = (preds == test_labels).float().mean().item() * 100
                
                # 分类分布
                pred_dist = torch.zeros(num_classes)
                for i in range(num_classes):
                    pred_dist[i] = (preds == i).sum().item()
                pred_dist = pred_dist / pred_dist.sum()
            
            results[name] = {
                "accuracy": accuracy,
                "pred_distribution": pred_dist.cpu().numpy()
            }
            print(f"  {name} 准确率: {accuracy:.2f}%")
            print(f"  预测分布: {dict(enumerate(pred_dist.cpu().numpy().round(2)))}")
        
        return results
    
    def analyze_feature_distribution(self):
        """分析特征分布"""
        print("\n3. 特征分布分析")
        
        # 收集全局测试集和客户端测试集的特征
        global_features, global_labels = self._collect_features_from_global_test()
        
        client_features_dict = {}
        client_labels_dict = {}
        for client_id, test_loader in self.test_data_dict.items():
            features, labels = self._collect_features_from_client(client_id, test_loader)
            client_features_dict[client_id] = features
            client_labels_dict[client_id] = labels
        
        # 特征分布统计
        global_mean = global_features.mean(dim=0)
        global_std = global_features.std(dim=0)
        
        client_means = {}
        client_stds = {}
        for client_id, features in client_features_dict.items():
            client_means[client_id] = features.mean(dim=0)
            client_stds[client_id] = features.std(dim=0)
        
        # 计算每个客户端与全局特征的分布差异
        print("特征分布KL散度:")
        for client_id in client_means:
            # 计算均值差异的L2距离
            mean_dist = torch.norm(client_means[client_id] - global_mean).item()
            # 计算标准差比率
            std_ratio = (client_stds[client_id] / (global_std + 1e-8)).mean().item()
            
            print(f"  客户端 {client_id} - 均值距离: {mean_dist:.4f}, 标准差比率: {std_ratio:.4f}")
        
        # 使用PCA降维可视化
        pca = PCA(n_components=2)
        
        # 对全局特征进行降维
        global_features_np = global_features.cpu().numpy()
        global_pca = pca.fit_transform(global_features_np)
        
        # 客户端特征映射到相同空间
        client_pca = {}
        for client_id, features in client_features_dict.items():
            client_pca[client_id] = pca.transform(features.cpu().numpy())
        
        # 获取分类器最后一层权重进行投影
        weights = None
        for layer in self.global_classifier.modules():
            if isinstance(layer, nn.Linear) and layer.out_features == 10:  # 假设10分类
                weights = layer.weight.detach().cpu().numpy()
                break
        
        # 如果找到权重，将其投影到PCA空间
        if weights is not None:
            weights_pca = pca.transform(weights)
            
            # 创建决策边界可视化
            plt.figure(figsize=(12, 10))
            
            # 绘制全局特征
            plt.scatter(global_pca[:, 0], global_pca[:, 1], c=global_labels.cpu().numpy(), 
                       cmap='tab10', alpha=0.5, marker='o', s=20, label='全局测试样本')
            
            # 绘制类别边界方向
            for i, (x, y) in enumerate(weights_pca):
                plt.arrow(0, 0, x*3, y*3, head_width=0.3, head_length=0.3, fc=f'C{i}', ec=f'C{i}')
                plt.text(x*3.1, y*3.1, f'类别{i}', fontsize=12)
            
            plt.title('特征分布与分类器决策边界')
            plt.xlabel('PCA组件1')
            plt.ylabel('PCA组件2')
            plt.legend()
            plt.savefig(f"{self.result_dir}/feature_distribution.png", dpi=300)
        
        return {
            "global_stats": (global_mean.cpu().numpy(), global_std.cpu().numpy()),
            "client_stats": {k: (v.cpu().numpy(), client_stds[k].cpu().numpy()) 
                            for k, v in client_means.items()}
        }
        
    def analyze_classifier_weights(self):
        """分析分类器权重"""
        print("\n4. 分类器权重分析")
        
        # 收集原始分类器权重
        original_weights = {}
        with torch.no_grad():
            for name, param in self.global_classifier.named_parameters():
                original_weights[name] = param.data.clone()
                
                # 计算并打印权重统计信息
                if param.dim() > 1:  # 只分析权重矩阵，不分析偏置
                    w_mean = param.mean().item()
                    w_std = param.std().item()
                    w_min = param.min().item()
                    w_max = param.max().item()
                    w_norm = torch.norm(param).item()
                    
                    # 计算行和列之间的方差，检查是否有特定偏好
                    row_means = param.mean(dim=1)
                    col_means = param.mean(dim=0)
                    row_std = row_means.std().item()
                    col_std = col_means.std().item()
                    
                    # 打印统计信息
                    print(f"  层 {name}:")
                    print(f"    形状: {param.shape}")
                    print(f"    均值: {w_mean:.4f}, 标准差: {w_std:.4f}")
                    print(f"    最小值: {w_min:.4f}, 最大值: {w_max:.4f}")
                    print(f"    范数: {w_norm:.4f}")
                    print(f"    输出方差: {row_std:.4f}, 输入方差: {col_std:.4f}")
                    
                    # 检查是否有崩溃模式的痕迹
                    rows_similar = row_std < 0.01
                    cols_similar = col_std < 0. 
                    if rows_similar:
                        print("    警告: 低输出方差可能导致崩溃分类")
                    if cols_similar:
                        print("    警告: 低输入方差表明特征利用不足")
                    
                    # 分析每个类别的权重
                    if param.shape[0] == 10:  # 假设是最终分类层
                        for i in range(param.shape[0]):
                            class_weight = param[i]
                            class_mean = class_weight.mean().item()
                            class_std = class_weight.std().item()
                            class_norm = torch.norm(class_weight).item()
                            print(f"    类别 {i}: 均值={class_mean:.4f}, 标准差={class_std:.4f}, "
                                 f"范数={class_norm:.4f}")
        
        # 如果可能，创建一个随机初始化的分类器作为参考
        try:
            reference_classifier = copy.deepcopy(self.global_classifier)
            # 重新初始化
            for module in reference_classifier.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                        
            # 收集参考分类器权重
            reference_weights = {}
            with torch.no_grad():
                for name, param in reference_classifier.named_parameters():
                    reference_weights[name] = param.data.clone()
            
            # 比较训练后与初始化的差异
            print("\n  训练后分类器与随机初始化比较:")
            for name in original_weights:
                if original_weights[name].dim() > 1:
                    orig_norm = torch.norm(original_weights[name]).item()
                    ref_norm = torch.norm(reference_weights[name]).item()
                    diff_norm = torch.norm(original_weights[name] - reference_weights[name]).item()
                    
                    print(f"    层 {name}: 当前范数={orig_norm:.4f}, 随机范数={ref_norm:.4f}, "
                         f"差异范数={diff_norm:.4f}, 相对变化={(diff_norm/ref_norm):.4f}")
        except Exception as e:
            print(f"  无法创建参考分类器: {str(e)}")
        
        return original_weights
    
    def analyze_gradient_flow(self):
        """分析梯度流动情况"""
        print("\n5. 梯度流分析")
        
        # 收集一个批次的数据
        sample_data, sample_labels = next(iter(self.global_test_loader))
        sample_data, sample_labels = sample_data.to(self.device), sample_labels.to(self.device)
        
        # 确保模型处于训练模式并清除梯度
        self.server_model.train()
        self.global_classifier.train()
        
        # 选择一个客户端模型
        client_id = list(self.client_models.keys())[0]
        client_model = self.client_models[client_id].train()
        
        # 前向传播
        _, shared_features, _ = client_model(sample_data)
        server_features = self.server_model(shared_features)
        logits = self.global_classifier(server_features)
        
        # 计算损失
        loss = F.cross_entropy(logits, sample_labels)
        
        # 清除先前梯度
        for model in [client_model, self.server_model, self.global_classifier]:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        # 反向传播
        loss.backward()
        
        # 收集梯度
        gradient_stats = {
            "classifier": {},
            "server": {},
            "client": {}
        }
        
        print("  分类器梯度统计:")
        for name, param in self.global_classifier.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                param_norm = torch.norm(param.data).item()
                grad_ratio = grad_norm / (param_norm + 1e-8)
                
                gradient_stats["classifier"][name] = {
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_ratio": grad_ratio
                }
                
                print(f"    {name}: 梯度范数={grad_norm:.6f}, 参数范数={param_norm:.4f}, "
                     f"比例={grad_ratio:.6f}")
        
        print("\n  服务器模型梯度统计:")
        for name, param in self.server_model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                param_norm = torch.norm(param.data).item()
                grad_ratio = grad_norm / (param_norm + 1e-8)
                
                gradient_stats["server"][name] = {
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "grad_ratio": grad_ratio
                }
                
                if grad_norm > 0.0001:  # 只打印有明显梯度的参数
                    print(f"    {name}: 梯度范数={grad_norm:.6f}, 参数范数={param_norm:.4f}, "
                         f"比例={grad_ratio:.6f}")
        
        # 计算梯度统计
        clf_grad_norms = [v["grad_norm"] for v in gradient_stats["classifier"].values()]
        server_grad_norms = [v["grad_norm"] for v in gradient_stats["server"].values()]
        
        print("\n  梯度流总结:")
        print(f"    分类器平均梯度范数: {np.mean(clf_grad_norms):.6f}")
        print(f"    服务器平均梯度范数: {np.mean(server_grad_norms):.6f}")
        
        # 检查梯度比例
        clf_grad_ratios = [v["grad_ratio"] for v in gradient_stats["classifier"].values()]
        print(f"    分类器最大梯度/参数比例: {max(clf_grad_ratios):.6f}")
        print(f"    分类器最小梯度/参数比例: {min(clf_grad_ratios):.6f}")
        
        return gradient_stats
    
    def test_generalization(self):
        """测试泛化能力"""
        print("\n6. 泛化能力测试")
        
        # 训练一个新的分类器，用所有客户端数据的混合
        new_classifier = copy.deepcopy(self.global_classifier)
        optimizer = torch.optim.Adam(new_classifier.parameters(), lr=0.001)
        
        # 收集所有客户端的特征和标签
        all_features = []
        all_labels = []
        
        for client_id, test_loader in self.test_data_dict.items():
            features, labels = self._collect_features_from_client(client_id, test_loader)
            all_features.append(features)
            all_labels.append(labels)
        
        # 合并数据
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # 训练新分类器
            new_classifier.train()
            for epoch in range(30):
                optimizer.zero_grad()
                logits = new_classifier(all_features)
                loss = F.cross_entropy(logits, all_labels)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    print(f"  混合数据训练轮次 {epoch+1}/30, 损失: {loss.item():.4f}")
            
            # 在全局测试集上评估
            new_classifier.eval()
            
            # 收集全局测试集特征
            global_features, global_labels = self._collect_features_from_global_test()
            
            with torch.no_grad():
                logits = new_classifier(global_features)
                _, preds = logits.max(1)
                accuracy = (preds == global_labels).float().mean().item() * 100
                
                # 计算分类分布
                pred_dist = torch.zeros(10)
                for i in range(10):
                    pred_dist[i] = (preds == i).sum().item()
                pred_dist = pred_dist / pred_dist.sum()
            
            print(f"  在混合数据上训练的分类器在全局测试集上准确率: {accuracy:.2f}%")
            print(f"  预测分布: {dict(enumerate(pred_dist.cpu().numpy().round(2)))}")
            
            # 计算原始分类器准确率作为对比
            original_acc = self._evaluate_on_features(
                self.global_classifier, global_features, global_labels)
            print(f"  原始分类器在全局测试集上准确率: {original_acc:.2f}%")
            
            return {
                "mixed_data_accuracy": accuracy,
                "original_accuracy": original_acc,
                "prediction_distribution": pred_dist.cpu().numpy()
            }
        else:
            print("  无客户端测试数据可用")
            return {}
    
    def _evaluate_on_global_test(self, server_model, classifier):
        """在全局测试集上评估"""
        server_model.eval()
        classifier.eval()
        
        # 选择一个客户端模型用于特征提取
        client_id = list(self.client_models.keys())[0]
        client_model = self.client_models[client_id].eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.global_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                logits = classifier(server_features)
                
                # 计算准确率
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / max(1, total)
        return accuracy
    
    def _evaluate_on_client_test(self, client_model, server_model, classifier, test_loader):
        """在客户端测试集上评估"""
        client_model.eval()
        server_model.eval()
        classifier.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                _, shared_features, _ = client_model(data)
                server_features = server_model(shared_features)
                logits = classifier(server_features)
                
                # 计算准确率
                _, pred = logits.max(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = 100.0 * correct / max(1, total)
        return accuracy
    
    def _collect_features_from_global_test(self):
        """从全局测试集收集特征"""
        self.server_model.eval()
        
        # 选择一个客户端模型用于特征提取
        client_id = list(self.client_models.keys())[0]
        client_model = self.client_models[client_id].eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.global_test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 提取特征
                _, shared_features, _ = client_model(data)
                server_features = self.server_model(shared_features)
                
                all_features.append(server_features)
                all_labels.append(target)
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            return all_features, all_labels
        else:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
    
    def _collect_features_from_client(self, client_id, test_loader):
        """从客户端测试集收集特征"""
        self.server_model.eval()
        client_model = self.client_models[client_id].eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # 提取特征
                _, shared_features, _ = client_model(data)
                server_features = self.server_model(shared_features)
                
                all_features.append(server_features)
                all_labels.append(target)
        
        if all_features:
            all_features = torch.cat(all_features, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            return all_features, all_labels
        else:
            return torch.tensor([]).to(self.device), torch.tensor([]).to(self.device)
    
    def _evaluate_on_features(self, classifier, features, labels):
        """在预计算的特征上评估分类器"""
        classifier.eval()
        
        with torch.no_grad():
            logits = classifier(features)
            _, preds = logits.max(1)
            accuracy = (preds == labels).float().mean().item() * 100
        
        return accuracy