import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class INP_Former(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder =[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token=None,
            embed_dim=None,  # 新增参数
    ) -> None:
        super(INP_Former, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token = prototype_token[0]

        # 如果 embed_dim 未提供，尝试从 prototype_token 推断
        if embed_dim is None:
            if self.prototype_token is not None:
                embed_dim = self.prototype_token.shape[-1]
            else:
                raise ValueError("embed_dim must be provided or prototype_token must exist")

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        # 移除 class token 和 register tokens
        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        # 计算特征图边长（基于 patch 数）
        side = int(math.sqrt(en_list[0].shape[1]))  # 假设为正方形

        # 融合多尺度特征
        x = self.fuse_feature(en_list)  # [B, N, C]
        # features = x  # 保存用于实例分类的特征

        # 原型聚合
        agg_prototype = self.prototype_token
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        g_loss = self.gather_loss(x, agg_prototype)

        # bottleneck
        for blk in self.bottleneck:
            x = blk(x)

        # decoder
        de_list = []
        for blk in self.decoder:
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]
        
        # 构建多尺度编码器和解码器特征
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        # 重塑为 [B, C, side, side] 的图像形式
        en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]

        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)