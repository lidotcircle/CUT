from torch import nn
from .modules import Attention, FeedForward, PreNorm


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x, layers = [], encode_only: bool = False):
        added = 0
        layerid = 0
        features = []
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            if layerid in layers:
                features.append(x)
                added = added + 1
            if encode_only and added == len(layers):
                break
            layerid = layerid + 1
        
        if len(layers) > 0:
            if encode_only:
                return features
            else:
                return x, features

        return x
