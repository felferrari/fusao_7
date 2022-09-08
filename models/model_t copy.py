import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dropout, GlobalAveragePooling1D, Dense, Conv2D
from tensorflow.keras import Model
from .models import resunet_encoder, resunet_decoder, resunet_classifier
from .transformer import  SwinTransformerBlock, PatchEmbed, BasicLayer, PatchMerging, Patch_expanding
import numpy as np


class SM_Transformer_PM(Model):
    def __init__(
        self, 
        n_classes,
        name = ''):

        super().__init__()
        self.encoder = SwinUnetEncoder()

        self.patch_expand = Patch_expanding(
            num_patch = (8,8), 
            embed_dim = 384, 
            upsample_rate = 2)
    
    
    def call(self, inputs):
        input_0 = inputs[0]
        input_1 = inputs[1]
        previous_input = inputs[2]

        input = tf.concat([input_0,  input_1, previous_input], axis=-1)

        x = self.encoder(input)

        return tf.random.normal(shape=(12, 128, 128, 3))

class SwinUnetEncoder(tf.keras.layers.Layer):
    def __init__(self, model_name='encoder', include_top=False,
                 img_size=(128, 128), patch_size=(4, 4), in_chans=27, num_classes=3,
                 embed_dim=96, depths=[2, 2, 2], num_heads=[3, 6, 12],
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=LayerNormalization, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__(name=model_name)

        self.include_top = include_top

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute postion embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(
                                                          1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build layers
        self.basic_layers = tf.keras.Sequential([BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                window_size=window_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                norm_layer=norm_layer,
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                use_checkpoint=use_checkpoint,
                                                prefix=f'layers{i_layer}') for i_layer in range(self.num_layers)])
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.avgpool = GlobalAveragePooling1D()
        if self.include_top:
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        #x = self.norm(x)
        #x = self.avgpool(x)
        return x

    def call(self, x):
        x = self.forward_features(x)
        #if self.include_top:
        #    x = self.head(x)
        return x