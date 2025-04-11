import timm
import torch
import torch.nn as nn
import math, json
from functools import reduce
from operator import mul
import clip
import torch.nn.functional as F


class SVDB_Transformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, args, **kwargs):
        # super().__init__(**kwargs)
        super(SVDB_Transformer, self).__init__(
            img_size=args.img_size, num_classes=args.num_classes)
        self.args = args
        self.num_layers = len(self.blocks)
        self.block_gradients = [None] * self.num_layers
        for i, block in enumerate(self.blocks):
            block.register_full_backward_hook(self.save_gradients(i))
        self.num_layers = len(self.blocks)

    def save_gradients(self, index):
        def hook(module, grad_input, grad_output):
            # Save gradients of the specific block
            self.block_gradients[index] = grad_output[0]

        return hook

    def clear_gradients(self):
        """Clears the stored gradients after usage."""
        self.block_gradients = [None] * self.num_layers

    def _pos_embed(self, x):  ##增加位置编码
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed

        return self.pos_drop(x)

    def forward_deep_prompt(self, x):
        hidden = []
        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.blocks[i](x)
                hidden.append(hidden_states)
            else:
                hidden_states = self.blocks[i](hidden_states)
                hidden.append(hidden_states)
        return hidden_states, hidden

    def forward_features(self, x):
        x = self.patch_embed(x)  # 把 x(64,224,224,3)patch ->>(64,198,768)
        x = self._pos_embed(x)  # 增加位置信息
        x, hidden = self.forward_deep_prompt(x)
        x = self.norm(x)
        return x, hidden

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:  # 使用cls token 而不使用 avg pool
            x = x[:, 0:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        semantic = x
        x = self.head(x)
        return semantic, x

    def forward(self, x):
        x, hidden = self.forward_features(x)
        semantic, x = self.forward_head(x)
        return x, semantic, hidden
