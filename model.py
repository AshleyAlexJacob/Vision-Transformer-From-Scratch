import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=8,
                 num_classes=10,
                 channels = 3,
                 dim = 128,
                 depth = 4, 
                 dropout = 0.1,
                 mlp_dim = 256,
                 heads = 4,
                 ):
        super(VisionTransformer, self).__init__()
        assert image_size%patch_size == 0, "Image size must be divisible by patch size."

        # No of patches

        self.num_patches = (image_size // patch_size)**2

        patch_dim = channels * patch_size * patch_size

        # patch embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # class token:  a learnable embedding that represents the whole image (added to sequence)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

        # postional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches+1, dim))

        # transformer encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead= heads,
            dim_feedforward= mlp_dim,
            dropout= dropout,
            activation= 'gelu',
            batch_first= True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer= encoder_layer,
            num_layers= depth
        )

        # classification head
        self.cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
    
    def forward(self, x):
         """
            x: input images of shape (batch, channels, height, width)
        """
         
         B,C,H,W = x.shape

         # 1. Divide image into patches and flatten
        
         patch_size = int(H // (H / (H // (H if H==W else 1))))  # quick way to get patch_size from H (here just use given patch_size)
        
          # Reshape into patches
         x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
         x = x.permute(0, 2, 4, 1, 3, 5)  # (B, n_patches_h, n_patches_w, C, patch_size, patch_size)
         patches = x.reshape(B, -1, C * patch_size * patch_size)  # (B, num_patches, patch_dim)
        
        # linear projection
         patch_embeddings = self.patch_to_embedding(patches)  # (B, num_patches, dim)

        #  add class token to patch embeddings
         cls_tokens = self.class_token.expand(B, -1, -1)
         x = torch.cat([cls_tokens, patch_embeddings], dim=1) # (B, num_patches + 1, dim)

        # Add postional embeddings
         x = x + self.positional_encoding[:, :x.size(1), :]

        #  pass to transformer encoder
         x = self.transformer(x) # (B, num_patches+1, dim)

        #  take the class token output, and apply classification head
         cls_output = x[:, 0]  # (B, dim)
         out = self.mlp_head(cls_output) # (B, num_classes)
         return out
    
if __name__ == "__main__":
    model = VisionTransformer(image_size=32, patch_size=4, num_classes=10, dim=128, depth=4, heads=4, mlp_dim=256)
