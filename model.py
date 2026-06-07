import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention layer.
    Allows different heads to attend to different parts of the sequence.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        
        # Key, Query, Value projections in a single batch projection
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to prevent attending to future tokens
        # Saved as buffer so it is moved to GPU if model is moved
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dimension (n_embd)

        # Calculate Query, Key, Value for all heads in batch
        # shape: [B, T, 3 * C]
        qkv = self.c_attn(x)
        # Split into query, key, and value vectors
        q, k, v = qkv.split(C, dim=2)

        # Reshape to [B, n_head, T, head_size]
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Causal self-attention: (Q @ K^T) / sqrt(head_size)
        # shape: [B, n_head, T, head_size] @ [B, n_head, head_size, T] -> [B, n_head, T, T]
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_size ** 0.5))
        
        # Apply causal masking: fill upper triangular parts with -inf
        # We slice bias down to current sequence length T
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        
        # Softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Compute weighted sum of values
        # shape: [B, n_head, T, T] @ [B, n_head, T, head_size] -> [B, n_head, T, head_size]
        y = att @ v
        
        # Reassemble all heads side-by-side: shape: [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity (GELU) and output projection.
    This performs token-wise computation.
    """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        # GPT-2 uses 4 * n_embd as the hidden dimension of the feedforward layer
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    A Transformer Block.
    Combines communication (MultiHeadAttention) and computation (FeedForward)
    with pre-layer normalization and residual connections.
    """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-LN residual connections (standard in modern Transformers)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """
    Custom Generative Pre-trained Transformer language model decoder from scratch.
    """
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=32, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Token and Position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        
        # Final LayerNorm before token prediction logits
        self.ln_f = nn.LayerNorm(n_embd)
        # Linear head projecting back to vocabulary size
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()

        # Token embeddings: [B, T, n_embd]
        tok_emb = self.token_embedding_table(idx)
        
        # Position embeddings: [B, T, n_embd]
        # Generate position indices (0, 1, ..., T-1) on the same device as idx
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding_table(pos) 
        
        # Combine embeddings
        x = tok_emb + pos_emb # shape: [B, T, n_embd]
        
        # Pass through Transformer blocks
        x = self.blocks(x) # shape: [B, T, n_embd]
        
        # Apply final LayerNorm
        x = self.ln_f(x) # shape: [B, T, n_embd]
        
        # Predict logits for next token: shape: [B, T, vocab_size]
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Reshape tensors to calculate CrossEntropy loss
            # logits: [B*T, vocab_size], targets: [B*T]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, stop_token_id=None):
        """
        Autoregressive generation.
        Generates new tokens, scaling logits by temperature and sampling from the distribution.
        """
        for _ in range(max_new_tokens):
            # Crop the context to fit our block size
            idx_cond = idx[:, -self.block_size:]
            
            # Forward pass to get logits for current sequence
            logits, _ = self(idx_cond)
            
            # Focus only on the last token's logits (the next prediction)
            # shape: [B, vocab_size]
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Break if stop token is reached
            if stop_token_id is not None and idx_next.item() == stop_token_id:
                break
                
        return idx
