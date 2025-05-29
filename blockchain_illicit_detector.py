import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc

# 设置随机种子保证可复现性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim 必须能被 num_heads 整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # 线性投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # 注意力加权和
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        return output

class FeaturePyramid(nn.Module):
    """时空特征金字塔 (步骤S4)"""
    def __init__(self, input_dim, meta_dim, embed_dim=128, num_heads=4):
        super().__init__()
        
        # 多模态融合层
        self.fusion = nn.Linear(input_dim + meta_dim, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        
        # 交易粒度层 (多头自注意力)
        self.trans_att = MultiHeadAttention(embed_dim, num_heads)
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_res = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        
        # 地址粒度层
        self.addr_dense1 = nn.Linear(embed_dim, embed_dim * 2)
        self.addr_dense2 = nn.Linear(embed_dim * 2, embed_dim)
        self.addr_norm = nn.LayerNorm(embed_dim)
        self.addr_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # 网络粒度层 (GRU)
        self.gru = nn.GRU(embed_dim, embed_dim // 2, batch_first=True, bidirectional=True)
        self.gru_norm = nn.LayerNorm(embed_dim)
        
        # 跨粒度融合
        self.pyramid_concat = nn.Linear(embed_dim * 3, embed_dim * 2)
        self.pyramid_norm = nn.LayerNorm(embed_dim * 2)
        
        # 时空门控机制
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.Sigmoid()
        )
        self.pyramid_output = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, original_features, meta_features):
        # 多模态融合
        fused = torch.cat([original_features, meta_features], dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_norm(fused)
        
        # ==== 交易粒度层 ====
        fused_seq = fused.unsqueeze(1)
        
        # 多头自注意力
        att_out = self.trans_att(fused_seq)
        att_out = self.trans_norm(att_out)
        
        # 残差连接
        trans_res = fused_seq + att_out
        trans_res = trans_res.squeeze(1)
        
        # 提取局部和全局特征
        trans_local = self.trans_res(trans_res)
        trans_global = self.trans_res(trans_res)
        trans_layer = torch.cat([trans_local, trans_global], dim=1)
        
        # ==== 地址粒度层 ====
        addr_layer = self.addr_dense1(trans_layer)
        addr_layer = self.addr_dense2(addr_layer)
        addr_layer = self.addr_norm(addr_layer)
        
        # 注意力门控
        gate_weights = self.addr_gate(addr_layer)
        addr_layer = addr_layer * gate_weights
        
        # ==== 网络粒度层 ====
        addr_seq = addr_layer.unsqueeze(1)
        gru_out, _ = self.gru(addr_seq)
        network_layer = gru_out.squeeze(1)
        network_layer = self.gru_norm(network_layer)
        
        # ==== 跨粒度特征融合 ====
        pyramid_features = torch.cat([trans_layer, addr_layer, network_layer], dim=1)
        pyramid_features = self.pyramid_concat(pyramid_features)
        pyramid_features = self.pyramid_norm(pyramid_features)
        
        # 时空门控机制
        gate_weights = self.gate(pyramid_features)
        gated_features = pyramid_features * gate_weights
        
        # 输出增强特征
        output = self.pyramid_output(gated_features)
        return output

class FinalClassifier(nn.Module):
    """最终分类器 (步骤S5)"""
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        
        # ==== 第一阶段: 基于Transformer的异常模式捕捉 ====
        self.transformer_att = MultiHeadAttention(input_dim, num_heads)
        self.transformer_norm = nn.LayerNorm(input_dim)
        self.transformer_gru = nn.GRU(input_dim, input_dim // 2, batch_first=True)
        
        # 快速推理路径
        self.fast_dense = nn.Sequential(
            nn.Linear(input_dim // 2, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.fast_output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # ==== 第二阶段: 深度分析路径 ====
        self.deep_dense1 = nn.Sequential(
            nn.Linear(input_dim // 2, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.deep_dense2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU()
        )
        self.deep_output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 动态路由门控
        self.confidence = nn.Sequential(
            nn.Linear(input_dim // 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pyramid_features):
        # 添加序列维度
        x = pyramid_features.unsqueeze(1)
        
        # Transformer编码器
        att_out = self.transformer_att(x)
        norm_out = self.transformer_norm(x + att_out)
        gru_out, _ = self.transformer_gru(norm_out)
        trans_enc = gru_out.squeeze(1)
        
        # 快速推理路径
        fast_path = self.fast_dense(trans_enc)
        fast_out = self.fast_output(fast_path)
        
        # 深度分析路径
        deep_path = self.deep_dense1(trans_enc)
        deep_path = self.deep_dense2(deep_path)
        deep_out = self.deep_output(deep_path)
        
        # 动态路由门控
        confidence = self.confidence(trans_enc)
        
        # 最终输出 (基于置信度选择路径)
        final_out = torch.where(
            confidence > 0.8,
            fast_out,
            deep_out
        )
        
        return fast_out, deep_out, confidence, final_out

class BlockchainFraudDetector:
    def __init__(self, n_base_models=15, subspace_ratio=0.5, n_classes=2, 
                 max_features=50, feature_names=None, embed_dim=128,
                 chunk_size=50000):
        """
        初始化非法交易检测器
        
        参数:
        n_base_models: 基分类器数量
        subspace_ratio: 子空间采样比例
        n_classes: 类别数量
        max_features: 最大特征数量
        feature_names: 特征名称列表
        embed_dim: 嵌入维度
        chunk_size: 分块处理大小
        """
        self.n_base_models = n_base_models
        self.subspace_ratio = subspace_ratio
        self.n_classes = n_classes
        self.max_features = max_features
        self.feature_names = feature_names
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        
        # 模型组件
        self.base_models = []  # 存储基模型和对应的特征索引
        self.feature_weights = None  # 特征采样权重
        self.attention_weights = None  # 注意力权重
        self.pyramid_model = None  # 特征金字塔模型
        self.classifier_model = None  # 最终分类器
        self.scaler = StandardScaler()  # 数据标准化器
        self.label_encoder = LabelEncoder()  # 标签编码器
        self.feature_importance = None  # 特征重要性
        self.selected_features = None  # 选择的特征索引
        
        # 训练数据缓存
        self.X_pool = None
        self.y_pool = None
        
    def preprocess_data(self, raw_data):
        """
        步骤S1: 数据清洗与特征工程
        
        参数:
        raw_data: 原始交易数据DataFrame
        
        返回:
        X: 特征矩阵
        y: 标签向量
        feature_names: 特征名称列表
        """
        # 数据清洗
        print("正在清洗数据...")
        cleaned_data = raw_data.copy()
        
        # 过滤无效数据
        cleaned_data = cleaned_data.dropna(subset=['label'])
        
        # 特征选择 - 根据实际数据集结构
        # 保留所有数值型特征，排除账户地址和标签
        exclude_cols = ['account', 'label']
        feature_cols = [col for col in cleaned_data.columns if col not in exclude_cols]
        
        # 标签处理
        y = cleaned_data['label'].values
        X = cleaned_data[feature_cols].values
        feature_names = feature_cols
        
        # 缓存特征名称
        self.feature_names = feature_names
        
        print(f"数据预处理完成，共提取 {len(feature_names)} 个特征")
        return X, y, feature_names
    
    def select_features(self, X, y, top_k=50):
        """
        特征选择 - 选择最重要的top_k个特征
        
        参数:
        X: 特征矩阵
        y: 标签向量
        top_k: 保留的特征数量
        """
        print(f"执行特征选择，保留最重要的 {top_k} 个特征...")
        
        # 使用随机森林进行特征重要性评估
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 获取特征重要性
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:top_k]
        
        # 更新特征集
        self.selected_features = indices
        X_reduced = X[:, indices]
        self.feature_names = [self.feature_names[i] for i in indices]
        
        print(f"特征选择完成，从 {X.shape[1]} 个特征中选择 {top_k} 个重要特征")
        return X_reduced
    
    def init_feature_weights(self, X, y):
        """
        初始化特征采样权重 (步骤S2)
        
        参数:
        X: 特征矩阵
        y: 标签向量
        """
        print("初始化特征采样权重...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # 计算特征重要性
        gini_importance = rf.feature_importances_
        self.feature_importance = gini_importance
        
        # 归一化
        self.feature_weights = gini_importance / np.sum(gini_importance)
        
        # 低频特征增强 (专利优化点)
        low_freq_mask = self.feature_weights < np.median(self.feature_weights)
        self.feature_weights[low_freq_mask] *= 1.2  # 增加低频特征权重
        self.feature_weights /= np.sum(self.feature_weights)  # 重新归一化
        
        print("特征权重初始化完成")
        
    def random_subspace_sampling(self, X):
        """
        随机子空间采样 (步骤S2)
        
        参数:
        X: 特征矩阵
        
        返回:
        selected_indices: 选中的特征索引
        """
        n_features = X.shape[1]
        subspace_size = max(1, int(n_features * self.subspace_ratio))
        
        # 轮盘赌选择特征
        selected_indices = np.random.choice(
            n_features, 
            size=subspace_size,
            replace=False,
            p=self.feature_weights
        )
        return selected_indices
    
    def train_base_models_chunked(self, X, y, val_ratio=0.1):
        """
        分块训练基分类器 (针对大数据集优化)
        
        参数:
        X: 特征矩阵
        y: 标签向量
        val_ratio: 验证集比例
        """
        print(f"开始分块训练 {self.n_base_models} 个基分类器...")
        start_time = time.time()
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, stratify=y, random_state=42
        )
        
        # 初始化特征权重
        self.init_feature_weights(X_train, y_train)
        self.base_models = []
        model_performance = []
        model_feature_indices = []
        
        # 分块处理训练数据
        n_chunks = int(np.ceil(len(X_train) / self.chunk_size))
        
        for i in range(self.n_base_models):
            # 随机子空间采样
            feat_indices = self.random_subspace_sampling(X_train)
            model_feature_indices.append(feat_indices)
            
            # 训练LightGBM基分类器 (使用分块数据)
            model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                num_leaves=31,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42+i,
                n_jobs=-1
            )
            
            # 分块训练
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * self.chunk_size
                end_idx = min((chunk_idx + 1) * self.chunk_size, len(X_train))
                
                X_chunk = X_train[start_idx:end_idx, feat_indices]
                y_chunk = y_train[start_idx:end_idx]
                
                if chunk_idx == 0:
                    model.fit(X_chunk, y_chunk)
                else:
                    model.fit(X_chunk, y_chunk, init_model=model)
                
                # 释放内存
                del X_chunk, y_chunk
                gc.collect()
            
            # 验证集性能评估
            val_pred = model.predict(X_val[:, feat_indices])
            val_f1 = f1_score(y_val, val_pred)
            model_performance.append(val_f1)
            self.base_models.append(model)
            
            # 动态调整特征权重 (专利核心创新点)
            if val_f1 > np.mean(model_performance):
                # 表现好的模型增加其对应特征的权重
                self.feature_weights[feat_indices] *= 1.1
            else:
                # 表现差的模型减少其对应特征的权重
                self.feature_weights[feat_indices] *= 0.9
                
            # 归一化权重
            self.feature_weights /= np.sum(self.feature_weights)
        
        # 存储模型特征索引
        self.model_feature_indices = model_feature_indices
        
        # 初始化注意力权重 (基于模型性能)
        self.attention_weights = np.array(model_performance)
        self.attention_weights /= np.sum(self.attention_weights)
        
        elapsed = time.time() - start_time
        print(f"基分类器训练完成，耗时: {elapsed:.2f}秒")
        print(f"平均验证集F1分数: {np.mean(model_performance):.4f}")
    
    def get_meta_features(self, X):
        """
        构建语义元特征矩阵 (步骤S3)
        
        参数:
        X: 特征矩阵
        
        返回:
        meta_features: 元特征矩阵
        """
        print("构建语义元特征矩阵...")
        n_samples = X.shape[0]
        meta_features = np.zeros((n_samples, self.n_base_models * self.n_classes))
        
        # 分块处理避免内存溢出
        n_chunks = int(np.ceil(n_samples / self.chunk_size))
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, n_samples)
            
            X_chunk = X[start_idx:end_idx]
            
            for i, model in enumerate(self.base_models):
                feat_indices = self.model_feature_indices[i]
                probas = model.predict_proba(X_chunk[:, feat_indices])
                
                start_col = i * self.n_classes
                end_col = (i+1) * self.n_classes
                meta_features[start_idx:end_idx, start_col:end_col] = probas
                
                # 应用注意力加权 (专利创新点)
                meta_features[start_idx:end_idx, start_col:end_col] *= self.attention_weights[i]
            
            # 释放内存
            del X_chunk
            gc.collect()
        
        # 降维处理 (可选)
        if meta_features.shape[1] > self.max_features:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.max_features)
            meta_features = pca.fit_transform(meta_features)
            print(f"元特征降维至 {self.max_features} 维")
        
        return meta_features
    
    def build_models(self, input_dim, meta_dim):
        """
        构建PyTorch模型 (特征金字塔和分类器)
        
        参数:
        input_dim: 原始特征维度
        meta_dim: 元特征维度
        """
        print("构建PyTorch模型...")
        # 特征金字塔模型
        self.pyramid_model = FeaturePyramid(input_dim, meta_dim, self.embed_dim).to(device)
        
        # 最终分类器
        self.classifier_model = FinalClassifier(self.embed_dim).to(device)
        
        print(f"特征金字塔输入维度: {input_dim}, 元特征维度: {meta_dim}")
        print(f"特征金字塔输出维度: {self.embed_dim}")
        
    def train_full_model(self, X, y, test_size=0.2, val_size=0.1, 
                         epochs=30, batch_size=1024, lr=0.001, top_k=50):
        """
        训练完整模型 (针对大数据集优化)
        
        参数:
        X: 特征矩阵
        y: 标签向量
        test_size: 测试集比例
        val_size: 验证集比例
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        top_k: 保留的特征数量
        """
        # 特征选择
        X = self.select_features(X, y, top_k)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), stratify=y_train, random_state=42
        )
        
        # 标准化特征
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # 缓存数据用于增量学习
        self.X_pool = np.vstack([X_train, X_val])
        self.y_pool = np.concatenate([y_train, y_val])
        
        # 步骤S2: 训练基分类器 (分块处理)
        self.train_base_models_chunked(X_train, y_train)
        
        # 步骤S3: 获取元特征 (分块处理)
        meta_train = self.get_meta_features(X_train)
        meta_val = self.get_meta_features(X_val)
        meta_test = self.get_meta_features(X_test)
        
        # 步骤S4: 构建PyTorch模型
        self.build_models(X_train.shape[1], meta_train.shape[1])
        
        # 转换为PyTorch张量
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        meta_train_t = torch.tensor(meta_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        meta_val_t = torch.tensor(meta_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_t, meta_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=4, pin_memory=True)
        
        val_dataset = TensorDataset(X_val_t, meta_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(
            list(self.pyramid_model.parameters()) + list(self.classifier_model.parameters()), 
            lr=lr,
            weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 训练循环
        best_val_loss = float('inf')
        train_losses, val_losses = [], []
        
        print("开始训练最终模型...")
        for epoch in range(epochs):
            self.pyramid_model.train()
            self.classifier_model.train()
            epoch_train_loss = 0.0
            
            # 训练批次
            for X_batch, meta_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                
                # 前向传播
                pyramid_out = self.pyramid_model(X_batch, meta_batch)
                _, _, _, final_out = self.classifier_model(pyramid_out)
                
                # 计算损失
                loss = criterion(final_out, y_batch)
                
                # 反向传播和优化
                loss.backward()
                nn.utils.clip_grad_norm_(self.pyramid_model.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.classifier_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item() * X_batch.size(0)
            
            # 计算平均训练损失
            epoch_train_loss /= len(train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            # 验证
            self.pyramid_model.eval()
            self.classifier_model.eval()
            epoch_val_loss = 0.0
            val_preds, val_targets = [], []
            
            with torch.no_grad():
                for X_batch, meta_batch, y_batch in val_loader:
                    pyramid_out = self.pyramid_model(X_batch, meta_batch)
                    _, _, _, final_out = self.classifier_model(pyramid_out)
                    
                    loss = criterion(final_out, y_batch)
                    epoch_val_loss += loss.item() * X_batch.size(0)
                    
                    val_preds.extend(final_out.cpu().numpy())
                    val_targets.extend(y_batch.cpu().numpy())
            
            # 计算平均验证损失
            epoch_val_loss /= len(val_loader.dataset)
            val_losses.append(epoch_val_loss)
            
            # 计算验证指标
            val_preds = np.array(val_preds).flatten()
            val_preds_binary = (val_preds > 0.5).astype(int)
            val_targets = np.array(val_targets).flatten()
            
            val_accuracy = np.mean(val_preds_binary == val_targets)
            val_f1 = f1_score(val_targets, val_preds_binary)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            # 学习率调整
            scheduler.step(epoch_val_loss)
            
            # 保存最佳模型
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save({
                    'pyramid_state_dict': self.pyramid_model.state_dict(),
                    'classifier_state_dict': self.classifier_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 'best_model.pth')
                print(f"保存最佳模型，验证损失: {best_val_loss:.4f}")
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # 加载最佳模型
        checkpoint = torch.load('best_model.pth')
        self.pyramid_model.load_state_dict(checkpoint['pyramid_state_dict'])
        self.classifier_model.load_state_dict(checkpoint['classifier_state_dict'])
        
        # 评估测试集
        test_preds, test_confidences, test_paths = self.predict(X_test)
        test_accuracy = np.mean((test_preds > 0.5) == y_test)
        test_f1 = f1_score(y_test, (test_preds > 0.5))
        test_auc = roc_auc_score(y_test, test_preds)
        
        print("\n模型训练完成")
        print(f"测试集准确率: {test_accuracy:.4f}, F1分数: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        return train_losses, val_losses
    
    def predict(self, X):
        """
        预测交易是否为非法
        
        参数:
        X: 特征矩阵
        
        返回:
        predictions: 预测概率
        confidences: 置信度
        paths: 使用的路径 ('fast' 或 'deep')
        """
        if self.pyramid_model is None or self.classifier_model is None:
            raise ValueError("模型尚未训练，请先调用 train_full_model 方法")
        
        # 标准化特征
        X = self.scaler.transform(X)
        
        # 如果进行了特征选择
        if self.selected_features is not None:
            X = X[:, self.selected_features]
        
        # 获取元特征
        meta = self.get_meta_features(X)
        
        # 转换为PyTorch张量
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        meta_t = torch.tensor(meta, dtype=torch.float32).to(device)
        
        # 预测
        self.pyramid_model.eval()
        self.classifier_model.eval()
        
        with torch.no_grad():
            # 获取金字塔特征
            pyramid_out = self.pyramid_model(X_t, meta_t)
            
            # 获取最终预测
            _, _, confidences, final_out = self.classifier_model(pyramid_out)
            
            # 转换为numpy数组
            predictions = final_out.cpu().numpy().flatten()
            confidences = confidences.cpu().numpy().flatten()
        
        # 确定使用的路径
        paths = ['fast' if c > 0.8 else 'deep' for c in confidences]
        
        return predictions, confidences, paths
    
    def incremental_learning(self, new_data):
        """
        增量学习机制 (针对大数据集优化)
        
        参数:
        new_data: 新增交易数据
        """
        print("执行增量学习...")
        start_time = time.time()
        
        # 预处理新数据
        X_new, y_new, _ = self.preprocess_data(new_data)
        
        # 如果进行了特征选择
        if self.selected_features is not None:
            X_new = X_new[:, self.selected_features]
        
        X_new = self.scaler.transform(X_new)
        
        # 扩展新特征组合
        new_models = []
        new_feature_indices = []
        
        for i in range(3):  # 为新增数据创建3个新基模型
            feat_indices = self.random_subspace_sampling(X_new)
            new_model = LGBMClassifier(n_estimators=50, random_state=100+i)
            new_model.fit(X_new[:, feat_indices], y_new)
            
            new_models.append(new_model)
            new_feature_indices.append(feat_indices)
        
        # 添加到基础模型
        self.base_models.extend(new_models)
        self.model_feature_indices.extend(new_feature_indices)
        
        # 更新注意力权重
        new_weights = np.ones(3) * np.mean(self.attention_weights)
        self.attention_weights = np.concatenate([self.attention_weights, new_weights])
        self.attention_weights /= np.sum(self.attention_weights)
        
        # 更新数据池
        if self.X_pool is None:
            self.X_pool = X_new
            self.y_pool = y_new
        else:
            self.X_pool = np.vstack([self.X_pool, X_new])
            self.y_pool = np.concatenate([self.y_pool, y_new])
        
        # 联合再训练 (只训练新模型)
        for i in range(len(self.base_models) - 3, len(self.base_models)):
            model = self.base_models[i]
            feat_indices = self.model_feature_indices[i]
            model.fit(self.X_pool[:, feat_indices], self.y_pool)
        
        # 微调金字塔模型
        self.fine_tune_pyramid(X_new, y_new)
        
        elapsed = time.time() - start_time
        print(f"增量学习完成，耗时: {elapsed:.2f}秒")
        print(f"当前基模型数量: {len(self.base_models)}")
    
    # ... (其余方法保持不变) ...

# =====================
# 示例用法 (针对真实数据集)
# =====================

def load_large_dataset(file_path, nrows=None, chunksize=50000):
    """加载大型数据集"""
    print(f"加载数据集: {file_path}")
    
    # 分块读取数据
    chunks = []
    for chunk in pd.read_excel(file_path, nrows=nrows, chunksize=chunksize):
        chunks.append(chunk)
    
    data = pd.concat(chunks, ignore_index=True)
    
    # 内存优化 - 转换数据类型
    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')
        elif data[col].dtype == 'int64':
            data[col] = data[col].astype('int32')
    
    print(f"数据集加载完成，共 {len(data)} 条记录")
    return data

if __name__ == "__main__":
    # 加载真实数据集
    data = load_large_dataset('工作簿1.xlsx')
    
    # 初始化检测器 (减少基模型数量以加速训练)
    detector = BlockchainFraudDetector(
        n_base_models=15, 
        subspace_ratio=0.6, 
        max_features=60, 
        embed_dim=128,
        chunk_size=50000
    )
    
    # 预处理数据
    X, y, feature_names = detector.preprocess_data(data)
    
    # 训练完整模型 (减少epochs并增加batch_size)
    train_losses, val_losses = detector.train_full_model(
        X, y, 
        epochs=20, 
        batch_size=4096, 
        lr=0.001,
        top_k=60  # 选择最重要的60个特征
    )
    
    # 评估模型
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    metrics = detector.evaluate_model(X_test, y_test)
    
    print("\n模型评估结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"快速路径使用率: {metrics['path_distribution']['fast'] / len(y_test):.2%}")
    print(f"深度路径使用率: {metrics['path_distribution']['deep'] / len(y_test):.2%}")
    
    # 可视化特征重要性
    detector.visualize_feature_importance(top_n=20)
    
    # 保存模型
    detector.save_model("blockchain_fraud_detector_real_data.pkl")