import torch
from torch import nn
import sys

from ..layers.Transformer_EncDec import Encoder, EncoderLayer
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from .layers.Embed import PatchEmbedding
from .models import PatchTST
from .models.LearningShapeletsSeg import LearningShapeletsSeg
from torch.nn import functional as F
import numpy


# 所以 ProtoPTST.Model 必须做到两件事：

# 像分类器一样输出 outputs（给 CrossEntropy 用）

# 像解释器一样输出 activations 和 prototype（给 prototype 学习、给 saliency 评分器用）


class Model(PatchTST.Model):
    
    
    def __init__(self, configs):
        super().__init__(configs)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding = configs.stride
        self.patch_num = (self.seq_len - self.patch_len + self.padding) // self.stride+1
        self.d_model = configs.d_model
        self.device= configs.device
        self.prototype_num_selected = 1
        
        self.num_prototypes = configs.num_prototypes
        self.num_classes = configs.num_class
        self.prototype_len = configs.prototype_len
        self.prototype_shape = [configs.num_prototypes,self.prototype_len,configs.enc_in] # [num_prototypes x d_model x enc_in]
        self.prototype_activation_function ='log'
        
        # 控制参数：是否使用 LearningShapeletsSeg（默认 False 使用原方式）
        self.use_shapelet_layer = getattr(configs, 'use_shapelet_layer', False)
        
        if self.use_shapelet_layer:
            # 使用 LearningShapeletsSeg 替代原来的 prototype_vectors + convolution_distance
            self.shapelet_layer = LearningShapeletsSeg(
                configs=configs,
                num_shapelets=configs.num_prototypes,
                shapelet_len=configs.prototype_len,
                in_channels=configs.enc_in,
                dist_measure=getattr(configs, 'dist_measure', 'euclidean'),
                znorm=getattr(configs, 'shapelet_znorm', True),
                temperature=getattr(configs, 'shapelet_temperature', 1.0),
                device=configs.device
            )
            
            print("使用 LearningShapeletsSeg 进行距离计算")
            print(f"距离度量: {self.shapelet_layer.cfg.dist_measure}")
            print(f"Z-归一化: {self.shapelet_layer.cfg.znorm}")
            print(f"温度参数: {self.shapelet_layer.cfg.temperature}")
        else:
            # 使用原来的方式：trainable prototype_vectors
            self.prototype_vectors_param = nn.Parameter(
                torch.randn(self.prototype_shape), 
                requires_grad=True
            )  # [num_prototypes, prototype_len, enc_in]
            
            # Xavier/Kaiming initialization
            if getattr(configs, 'prototype_init', 'kaiming') == 'kaiming':
                nn.init.kaiming_uniform_(self.prototype_vectors_param, a=0, mode='fan_in', nonlinearity='leaky_relu')
            else:
                nn.init.xavier_uniform_(self.prototype_vectors_param)
            
            print("使用原始 PatchTST 方式进行距离计算 (conv1d)")
            
        
        # Xavier initialization
        print("self.seq_len",self.seq_len)
        
        # 它的意图是：
        # 模型先算一个 “匹配曲线” activations，形状是 [bs, seq_len, num_prototypes]
        # 然后把它变成一个长向量 [bs, num_prototypes*(seq_len-1)]
        # 最后用线性层做分类。projection 这块不是“理论最优”，是为了让工程跑起来、维度永远能对上。


        # Compute expected input dimension
        expected_input_dim = (self.seq_len-1) * self.num_prototypes
        
        # Add adaptive pooling to handle dynamic-length input
        self.adaptive_pool = nn.AdaptiveAvgPool1d(expected_input_dim // self.num_prototypes)
        
        self.projection = nn.Linear(
                expected_input_dim, configs.num_class  #### !!!
            )
        
        self.projection_1= nn.Linear(
                 self.num_prototypes*101, configs.num_class)
        
        self.linear_1 = nn.Linear(128, 1)
        self.args = configs
        
        print("!!!!prototype_shape",self.prototype_shape)
        print("!!!!self.patch_num",self.patch_num)
        print("configs.d_model",configs.d_model)
        print("!!!!expected_input_dim", expected_input_dim)
        #作者不想让 prototype 只是“原始波形”，而是希望它也经过和输入序列相似的编码/变换，变成一种“更适合匹配”的表示。    
    
    @property
    def prototype_vectors(self):
        """向后兼容：根据模式返回对应的 prototype vectors"""
        if self.use_shapelet_layer:
            return self.shapelet_layer.get_shapelets()  # [P, L, C]
        else:
            return self.prototype_vectors_param  # [P, L, C]
    
    def prototype_layer(self, prototype, prototype_patch=True, if_normalize=False):
        """
        prototyes: [num_prototypes x prototype_len x enc_in]: torch.tensor
        可选归一化（零均值/单位方差）

        对 prototype 做 patching + embedding

        送进 encoder，最后用 linear_1 压到 1 维
        """
        # Normalization from Non-stationary Transformer
        if if_normalize:
            means = prototype.mean(dim=1, keepdim=True).detach()
            prototype = prototype - means
            stdev = prototype.std(dim=1, keepdim=True, unbiased=False) + 1e-5
            prototype = prototype / stdev 
            
        
        prototype_enc=prototype # [num_prototypes, prototype_len, enc_in]
        ##print("!!1!enc_out.shape",prototype_enc.shape)# 
        
        if prototype_patch:
            # do patching and embedding
            prototype_enc = prototype_enc.permute(0, 2, 1) # [num_prototypes,enc_in, prototype_len]
            ##print("!!1.5!enc_out.shape",prototype_enc.shape)
            prototype_out, n_vars = self.patch_embedding(prototype_enc)# [num_prototypes, patch_num, d_model]
            
            ##print("!!2!enc_out.shape",prototype_out.shape)

            #  Encoder
            # z: [bs * nvars x patch_num x d_model]
            prototype_out, attns = self.encoder(prototype_out) # [num_prototypes, patch_num, d_model]
            ##print("!!3!prototype.shape",prototype_out.shape)
            prototype_out=self.linear_1(prototype_out)  # [num_prototypes, patch_num, 1]
    
        
            #prototype=self.prototype_vectors # !!! test
            
            ##print("!!4!prototype.shape",prototype_out.shape)
        
                
            
        
        return prototype_out
        
#
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        input:
        x_enc: [bs x seq_len x nvars ]
        
        output: [bs x num_classes]
        """
        x_enc=x_enc+2.5### ！ for freaqshape
        
        x=x_enc # ([4, 1400, 1])
        #print("x.shape",x.shape)
        
        if self.use_shapelet_layer:
            # 使用 LearningShapeletsSeg 计算 activations
            activations = self.shapelet_layer(x)  # [bs, seq_len, num_prototypes]
            
            # 获取当前的 shapelets（用于 loss 计算和可视化）
            prototype = self.shapelet_layer.get_shapelets()  # [P, L, C]
        else:
            # 使用原来的方式：prototype_layer + convolution_distance
            if self.args.ablation != 'no_prototype_layer':
                prototype = self.prototype_layer(
                    self.prototype_vectors_param, 
                    prototype_patch=True, 
                    if_normalize=False
                )  # [num_prototypes, patch_num, 1]
            else:
                prototype = self.prototype_vectors_param  # [num_prototypes, prototype_len, enc_in]
            
            # 计算 activations
            activations = self.convolution_distance_x_as_input(x, prototype)  # [bs, seq_len, num_prototypes]
        
        # Reshape to [batch_size, num_prototypes, seq_len] for 1D adaptive pooling
        activations_for_pool = activations.permute(0, 2, 1)  # [bs x num_prototypes x seq_len]
        
        # Use adaptive pooling to standardize sequence length
        expected_seq_len = (self.seq_len-1)
        
        # Apply adaptive pooling
        pooled_activations = self.adaptive_pool(activations_for_pool)  # [bs x num_prototypes x expected_seq_len]
        #把“匹配曲线”统一到固定长度（否则分类头尺寸对不上）
        # Reshape to a flat vector
        output = pooled_activations.reshape(pooled_activations.shape[0], -1)  # [bs x (num_prototypes * expected_seq_len)]
        #把多 prototype 的曲线拼成一个固定维度向量（否则 Linear 不能直接吃 3D）
        if False:
            indices=self.prototype_selection(activations,prototype) #[bs x n_selected]
            indices= indices.unsqueeze(-1).expand(-1, -1, self.seq_len) # #[bs x n_selected,seq_len]
            indices=indices.permute(0,2,1) # [bs x seq_len x n_selected]
            output= torch.gather(activations, dim=2, index=indices).reshape(activations.shape[0],-1)

        
        # Dimension adaptation (fallback)
        expected_dim = self.projection.weight.shape[1]
        current_dim = output.shape[1]
        
        if current_dim != expected_dim:
            print(f"Warning: Dimension mismatch even after pooling. Adapting input from {current_dim} to {expected_dim}")
            
            if current_dim > expected_dim:
                # Truncate if input dimension is too large
                output = output[:, :expected_dim]
            else:
                # Pad with zeros if input dimension is too small
                padding = torch.zeros(output.shape[0], expected_dim - current_dim, device=output.device)
                output = torch.cat([output, padding], dim=1)
        
        output = self.projection(output)
        
        #output=self.projection_1(x.reshape(x.shape[0],-1)) # test
    #         logits：给分类 loss

    # activations：告诉你“哪里匹配哪个 prototype” → 用于解释/也用于 prototype loss 抽片段

    # prototype：当前 prototype 的实际形状 → 用于 loss 或可视化
        return output, activations, prototype
        
          
    
    def convolution_distance_x_as_input(self, x, prototype_vectors):
        """
        x: [bs x patch_num x nvars]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        
        x=x.permute(0,2,1) # [bs, nvars, patch_num  ]
        
  
        prototype_vectors = prototype_vectors.permute(0, 2, 1)  # [num_prototypes, 1 , porto_len]
        prototype_vectors=prototype_vectors.to(x.device)
        # Perform convolution
        ##print("In convolution_distance, x.shape",x.shape,"prototype_vectors.shape",prototype_vectors.shape) # devie is x.device
        
        conv_out = F.conv1d(x, prototype_vectors, padding=(prototype_vectors.shape[-1]-1)//2 ,stride=1)  # 输出 [bs, num_prototypes, patch_num]
        
        # Adjust output shape and apply normalization
        conv_out = conv_out.permute(0, 2, 1)  # [bs, patch_num, num_prototypes]
        #输入序列和每个 prototype 的滑动相关/匹配强度
        
        # layer_norm = nn.LayerNorm(conv_out.shape[1:]).to(self.device)  # LayerNorm based on conv_out shape
        # #对每个样本，把 [T, P] 这个二维块整体做归一化（均值 0，方差 1，外加可学习的 scale/bias）。
        # #把“匹配强度”变成一个相对尺度，更关注“哪里相对更强”，而不是绝对值有多大
        # conv_out = layer_norm(conv_out)

        conv_out_tlast = conv_out.permute(0, 2, 1)  # [bs, P, T]
        layer_norm = nn.LayerNorm(conv_out_tlast.shape[-1]).to(conv_out.device)  # 只 LN 时间维 T
        conv_out_tlast = layer_norm(conv_out_tlast)
        conv_out = conv_out_tlast.permute(0, 2, 1)  # [bs, T, P]        



        # conv_out = torch.sigmoid(conv_out)
        conv_out=F.leaky_relu(conv_out, negative_slope=0.01) ## !!!
        conv_out=F.leaky_relu(conv_out, negative_slope=0.0001) ## !!!
        #正区间不变（保留强匹配）；负区间保留一个小斜率（负值仍能传梯度，不至于“死掉”）
        ##print("conv_out.shape",conv_out.shape) # [bs x patch_num x num_prototypes]

        return conv_out

  
    def smoothness_loss(self,sequence):
        """
        not in use
        """
        
        sequence=sequence.reshape(-1)
        # sequence shape: [batch_size, sequence_length]
        diff = sequence[:1:] - sequence[:-1]  # Compute differences between adjacent elements
        return torch.mean(diff ** 2)  # Square differences and take the mean
    
    
    def seg_prototype_loss(self,activations,x,prototype_vectors,outputs,select_prototype=True,no_variance_loss=False):
        
        """
        如果你只做分类 loss（CrossEntropy），prototype 可能学成“奇怪的东西”：
        只要能让分类正确，它不一定像“真实片段”。
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        if self.args.ablation == 'no_variances_loss':
            no_variance_loss = True
        
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       
        half_len = prototype_len // 2
        x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # Use torch.topk to get top-1 max value and its index per sample
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # Placeholder for per-sample subsequences (zeros)


            # Initialize a tensor to store extracted subsequences
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
#             对每个样本、每个 prototype：

# 在 activation 曲线上找最大位置 center

# 从原始序列截 [center-half_len : center+half_len]

# 越界就前/后补零到固定长度 prototype_len

# 减去均值（零均值化）

            # Extract subsequence around the max index for each sample
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # center position
                # Compute subsequence start/end within bounds
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # Pad to prototype_len at sequence boundaries if needed
                sub_seq = x[i, start:end]
                
                if len(sub_seq) < prototype_len:
                    # Pad at the beginning if start == 0
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # Otherwise pad at the end
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                
                # Store into subsequences
                mean = sub_seq.mean()
                # Normalize to zero-mean
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,0,2) # [ bs x num_prototypes x prototype_len]
        
        prototype_repeat=prototype_vectors.permute(2,0,1) #  [1,num_prototypes, patch_num ]
        prototype_repeat=prototype_repeat.repeat(batch_size, 1,1)
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        
        # if select_prototype and self.num_prototypes>1:
        if False:
            
            #indices=self.prototype_selection(activations,prototype_vectors) #[bs x n_selected]
            
            indices=self.prototype_selection_class(outputs,prototype_vectors) #[bs x 1]
            # print("############indices",indices)
            
            indices= indices.unsqueeze(-1).expand(-1, -1, prototype_len) #
            
            #print("indices.shape",indices[:,0,0])
            
            
       
            
            
            subsequence_all_prototype= torch.gather(subsequence_all_prototype, dim=1, index=indices)
            prototype_repeat= torch.gather(prototype_repeat, dim=1, index=indices)
            
            #print("subsequence_all_prototype.shape",subsequence_all_prototype.shape)
            
        
        #print("subsequence_all_prototype.shape",prototype_repeat)
        
        prototype_loss = torch.nn.functional.mse_loss(subsequence_all_prototype,prototype_repeat)
        #print("@@@prototype_loss",prototype_loss)
        variance_loss = self.total_variance_loss(prototype_vectors)
        
        
        if  self.args.ablation == 'no_matching_loss':
            prototype_loss = 0
        
        
        #print("proto_loss:",prototype_loss,"variance_loss:",variance_loss)
        if self.num_prototypes==1:
            return  prototype_loss
        
        if no_variance_loss:
            return prototype_loss
        else:
            return self.dynamic_loss(prototype_loss,variance_loss)

  
    def seg_activations(self,activations,prototype_vectors):
        
        """
        对每个 prototype，在每个样本里找到它最强匹配的位置，然后把该 prototype 的“激活曲线”在这个位置附近截取一段固定长度（prototype_len），作为一个局部激活片段。
        activatetion:  [bs x seq_len x num_prototypes]
        prototype_vectors: [num_prototypes, patch_num, 1]
        """
        
        batch_size = activations.shape[0]
        prototype_len = prototype_vectors.shape[1]
        prototype_num = self.num_prototypes
       #每个 prototype 真正“命中”的那一小段附近的激活形状
        half_len = prototype_len // 2
        #x=x.reshape(batch_size,-1)
        
        subsequence_all_prototype = torch.tensor([]).to(self.device)
        for p_index in range(prototype_num):
            ##print("p_index",p_index)
            # Use torch.topk to get top-1 max value and its index per sample
            activation = activations[:,:,p_index].reshape(batch_size, -1)
            
            #get the top values and indices
            ##print("activations.shape",activation.shape)
            top_k_values, top_k_indices = torch.topk(activation, 1, dim=1)  # [bs x 1]
            # Placeholder for per-sample subsequences (zeros)


            # Initialize a tensor to store extracted subsequences
            subsequences = torch.zeros(batch_size,prototype_len).to(self.device)
            

            # Extract subsequence around the max index for each sample
            for i in range(batch_size):
                center = top_k_indices[i, 0].item()  # center position
                # Compute subsequence start/end within bounds
                start = max(0, center - half_len)
                end = min(self.seq_len, center + half_len)
                
                # Pad to prototype_len at sequence boundaries if needed
                sub_seq =  activations[i,start:end,p_index]  #x[i, start:end] !!!
                #边界不足时补零
                if len(sub_seq) < prototype_len:
                    # Pad at the beginning if start == 0
                    if start == 0:
                        sub_seq = torch.cat((torch.zeros(prototype_len - len(sub_seq)).to(self.device), sub_seq))
                    # Otherwise pad at the end
                    else:
                        sub_seq = torch.cat((sub_seq, torch.zeros(prototype_len - len(sub_seq)).to(self.device)))
                #对截出的激活片段做零均值（中心化）
                # Store into subsequences
                mean = sub_seq.mean()
    #                 激活的绝对值会受 LayerNorm、数据分布等影响

    # 减均值后更关注“形状”：尖峰/宽峰/波动模式，而不是整体抬高还是压低
                # Normalize to zero-mean
                sub_seq = sub_seq - mean
                
                subsequences[i] = sub_seq
            
            subsequences=subsequences.unsqueeze(0) # [1 x bs x prototype_len] 每个 prototype 得到一个 subsequences: [bs, prototype_len]
            
            subsequence_all_prototype=torch.cat((subsequence_all_prototype,subsequences),dim=0) # [num_prototypes x bs x prototype_len]
            
        
        subsequence_all_prototype=subsequence_all_prototype.permute(1,2,0) # [ bs x  prototype_len x num_prototypes]
        
#        seg_prototype_loss：用峰值位置去切 原始 x 的子序列，让 prototype 像真实片段
# 输出是一个 loss（监督 prototype 学习）

# seg_activations：用峰值位置去切 activations 本身 的局部窗口
# 输出是一个张量（局部激活片段），更像是“分析/可视化用工具函数”
        ##print("subsequence_all_prototype.shape",subsequence_all_prototype.shape,"prototype_vectors.shape",prototype_vectors.shape)
        
        return subsequence_all_prototype
    
        
        
        


  
  
  
  
    def prototype_selection(self,conv_out,prototype_vectors):
        """
        conv_out: [bs x patch_num x num_prototypes]
        prototype_vectors: [num_prototypes, porto_len, 1]
        """
        # Select top-n prototypes
        n_selected = self.prototype_num_selected
        
        top_k_values, _ = torch.topk(conv_out, 1, dim=1) # [bs x 1 x num_prototypes]
        
        top_k_values, indices = torch.topk(top_k_values, n_selected, dim=2) # [bs x 1 x n_selected]
        
        indices = indices.squeeze(1) # [bs x n_selected]
        
        #selected_prototypes = torch.gather(conv_out, 1, indices)

        # selected_prototypes has shape [batch_size, n_selected, proto_len]
        
        #print('indices:',indices)
        return indices #[bs x n_selected]
    
    
    def prototype_selection_class(self,output,prototype_vectors):
        """
        output: [bs x num_classes]
        prototype_vectors: [num_prototypes, porto_len, 1]
        
        """        
        
        n_selected = self.prototype_num_selected
        
        indices = torch.argmax(output, dim=1) # [bs]
        
        indices = indices.unsqueeze(-1)
        
        return indices #[bs x 1]
        




    def total_variance_loss(self,prototype_vectors):
        """
            prototype_vectors: [num_prototypes, patch_num, 1]
            """
        prototype_vectors=prototype_vectors.reshape(prototype_vectors.shape[0],-1)
        
        
        
        mean_embedding = torch.mean(prototype_vectors, dim=0, keepdim=True)
        deviations = prototype_vectors - mean_embedding
        variance = torch.mean(torch.sum(deviations ** 2, dim=1))
        # Maximize variance (i.e., minimize negative variance)
        loss = -torch.log(variance)
        
        return loss
    
    def dynamic_loss(self,loss1,loss2):
        #）自动“按当前大小动态加权”组合成一个总 loss，避免某一个 loss 数值太大把另一个完全淹没。
        epsilon = 1e-5
        
        loss1=abs(loss1)
        loss2=abs(loss2)

        # Compute weights using inverse of loss values
        w1 = 1.0 / (loss1 + epsilon)
        w2 = 1.0 / (loss2 + epsilon)

        # Normalize weights
        sum_w = w1 + w2
        w1 = w1 / sum_w
        w2 = w2 / sum_w

        # Compute total loss
        
        normalized_loss1 = w1 * loss1
        normalized_loss2 = w2 * loss2
        
        #print("normalized_loss1",normalized_loss1,"normalized_loss2",normalized_loss2)
        total_loss = normalized_loss1 + normalized_loss2
        
        return total_loss
    
    def activation_cosine_loss(self,activations):
        """
        activations: [bs x seq_len x num_prototypes]
        """
        # If only one prototype, return 0 loss
        if self.num_prototypes==1:
            return 0
        
        else:
        
            X=activations[:,:,0]
            Y=activations[:,:,1]
            
            cosine_sim = F.cosine_similarity(X, Y, dim=1)
            
            #print("mean",mean,"variance",variance)
            # Compute loss on activations
            loss = 1 - cosine_sim.mean()
            return loss
    
    
    
    def interpolate_tensor(input_tensor,new_T=480,dim_to_change=0):

        # Swap dims to move T to last, easier for interpolate
        last_dim=len(input_tensor.shape)-1
        input_tensor = input_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, D, T)

        # Use interpolate
        output_tensor = torch.nn.functional.interpolate(input_tensor, size=new_T, mode='linear', align_corners=True)
        print(output_tensor.shape)

        # Swap dims back to original order
        output_tensor = output_tensor.transpose(dim_to_change,last_dim)  # Now (batch_size, new_T, D)

        return output_tensor