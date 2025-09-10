"""
Implementation inspired by the following paper, with modifications to support n modalities in 2D space instead of just two (text and image).
Liu, C., Ding, H., Zhang, Y., & Jiang, X. (2023). Multi-Modal Mutual Attention and Iterative Interaction for Referring Image Segmentation. IEEE Transactions on Image Processing, 32, 3054–3065. doi:10.1109/tip.2023.3277791
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiModalMutualAttention(nn.Module):
    """The proposed Multi-Modal Mutual Attention (M3Att) mechanism implemented from scratch"""

    def __init__(self, d_model, n_heads, nb_modality, dropout=0.1):
        super(MultiModalMutualAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = [nn.Linear(d_model, d_model) for _ in range(nb_modality)]
        self.w_k = [nn.Linear(d_model, d_model) for _ in range(nb_modality)]
        self.w_v = [nn.Linear(d_model, d_model) for _ in range(nb_modality)]
        self.w_o = nn.Linear(197, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.d_k)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention"""
        # Q, K, V shape: (batch_size, n_heads, seq_len, d_k)
        scores = self.mutual_attention(Q, K, V, mask)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # attention_weights = F.softmax(scores, dim=-1)

        # in the paper
        # attWeightsMod1 = softmax(scores )valuesMod1
        # attWeightsMod2 = softmax(scores T )valuesMod2
        # scores are transposed for the second modality because in the original paper, the second modality is text and the authors use the transposed attention scores for text
        # so we are assuming here every modality is a 2d input
        attentions_weights = [F.softmax(score, dim=-1) for score in scores]

        feature_outputs = []
        for i in range(len(attentions_weights)):
            feature_outputs.append(torch.matmul(attentions_weights[i], V[i]))
            print(f"Feature output[{i}] shape: {feature_outputs[i].shape}")
        
        # in the paper:
        # OutputMultimodal = featureMod1 (featureMod2 )T
        # in the case of more than 2 modalities, we can compute the output as the sum of all the pairwise products
        outputs = []
        for i in range(len(feature_outputs)):
            for j in range(len(feature_outputs)):
                if i != j:
                    outputs.append(torch.matmul(feature_outputs[i], feature_outputs[j].transpose(-2, -1)) * self.scale)
        output = sum(outputs) / len(outputs)  # average the outputs to keep the scale consistent
        print(f"Output shape: {output.shape}")

        return output, attentions_weights

    def mutual_attention(self, Q, K, V, mask=None):
        """
        in the paper, the mutual attention is defined as:
        Amut = (1/√C) * KeyMod1 ⋅ (KeyMod2 Transpose) 
        so for n modalities, few options are possible:
        - compute n*(n-1) mutual attention matrices
        - compute n-1 mutual attention matrices by chaining the keys
        - if there is a main modality, compute the mutual attention between this main modality and all the others
        - ...

        """
        outputs = []
        
        # first option
        for i in range(len(K)):
            for j in range(len(K)):
                if i != j:
                    outputs.append(torch.matmul(K[i], K[j].transpose(-2, -1)) * self.scale)

        # second options
        # for i in range(1, len(K)):
        #     outputs.append(torch.matmul(K[i - 1], K[i].transpose(-2, -1)) * self.scale)
            
        # third option
        # if len(K) > 0:
        #     main_modality = K[0] # assuming the first modality is the main one
        #     for i in range(1, len(K)):
        #         outputs.append(torch.matmul(main_modality, K[i].transpose(-2, -1)) * self.scale)

        return outputs

    def forward(self, x: list, mask=None):
        batch_sizes = []
        seq_lens = []
        d_models = []
        
        for modality in x:
            batch_size, seq_len, d_model = modality.size()
            batch_sizes.append(batch_size)
            seq_lens.append(seq_len)
            d_models.append(d_model)

        queries = [x_mod for x_mod in x]
        keys = [x_mod for x_mod in x]
        values = [x_mod for x_mod in x]

        Q = []
        K = []
        V = []
        for i in range(len(x)):
            Q.append(self.w_q[i](queries[i]).view(batch_sizes[i], seq_lens[i], self.n_heads, self.d_k).transpose(1, 2))
            K.append(self.w_k[i](keys[i]).view(batch_sizes[i], seq_lens[i], self.n_heads, self.d_k).transpose(1, 2))
            V.append(self.w_v[i](values[i]).view(batch_sizes[i], seq_lens[i], self.n_heads, self.d_k).transpose(1, 2))

        print(f"Q[0] shape: {Q[0].shape}, K[0] shape: {K[0].shape}, V[0] shape: {V[0].shape}")

        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        # attention_output = attention_output.transpose(1, 2).contiguous().view(
            # batch_size, seq_len, d_model
        # )

        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output, attention_weights

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_seq_length=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: list):
        # return x + self.pe[:, :x.size(1)]
        return [xi + self.pe[:, :xi.size(1)] for xi in x]
    

class TransformerBlock(nn.Module):
    """Single transformer block with attention and feed-forward"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, nb_modality=2):
        super(TransformerBlock, self).__init__()

        self.attention = MultiModalMutualAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, nb_modality=nb_modality)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: list, mask=None):
        attn_output, attention_weights = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attention_weights

class ImagePatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, d_model=768, nb_modality=2):
        super(ImagePatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.projections = [nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size) for _ in range(nb_modality)]
        
    def forward(self, x: list):
        # each value in x as shape of: (batch_size, channels, height, width)
        x = [proj(x_i) for proj, x_i in zip(self.projections, x)]
        x = [xi.flatten(2) for xi in x]
        x = [xi.transpose(1, 2) for xi in x] # list of (batch_size, n_patches, d_model)
        return x

class DIYTransformerForImageM3Att(nn.Module):
    """Vision Transformer with attention mechanism implemented from scratch"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, 
                 d_model=768, n_heads=12, n_layers=12, d_ff=3072, dropout=0.1, nb_modality=2):
        super(DIYTransformerForImageM3Att, self).__init__()
        
        self.patch_embedding = ImagePatchEmbedding(img_size, patch_size, in_channels, d_model, nb_modality)
        self.positional_encoding = PositionalEncoding(d_model, self.patch_embedding.n_patches + 1)
        
        # Class token
        self.cls_token = [nn.Parameter(torch.randn(1, 1, d_model)) for _ in range(nb_modality)]

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout=dropout, nb_modality=nb_modality)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following ViT paper"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
        
        # Initialize class token
        for cls_token in self.cls_token:
            torch.nn.init.normal_(cls_token, std=0.02)

    def forward(self, x: list):
        if not isinstance(x, list):
            raise ValueError("Input x must be a list of modalities")
        elif len(x) == 0:
            raise ValueError("Input list x is empty")

        batch_size = x[0].shape[0]
        
        # Convert image to patches and embed
        x = self.patch_embedding(x)  # (batch_size, n_patches, d_model)
        
        # Add class token
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = [self.cls_token[i].expand(batch_size, -1, -1) for i in range(len(x))]
        # x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, n_patches + 1, d_model)
        x = [torch.cat([cls_tokens[i], x[i]], dim=1) for i in range(len(x))]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        # x = self.dropout(x)
        x = [self.dropout(xi) for xi in x]
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            attention_weights.append(attn_weights)
        
        # Classification using class token
        x = self.norm(x)
        cls_output = x[:, 0]  # Use class token for classification
        output = self.classifier(cls_output)
        
        # return output, attention_weights
        return output

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = DIYTransformerForImageM3Att(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
        nb_modality=2
    )
    
    # Test with random input
    # x = torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224
    x = [torch.randn(2, 3, 224, 224) for _ in range(2)]  # batch_size=2, channels=3, height=224, width=224

    with torch.no_grad():
        output = model(x)  # Remove attention_weights since forward doesn't return them
        print(f"Input shape: {[xi.shape for xi in x]}")
        print(f"Output shape: {output.shape}")
        
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

