"""
MIDI Transformer Model

Conditional autoregressive transformer for MIDI generation.
Supports tempo, instrument, and mood conditioning.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class MIDITransformerConfig:
    """Configuration for MIDI Transformer model."""
    
    # Model architecture
    vocab_size: int = 512
    max_seq_len: int = 2048
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    
    # Conditioning
    n_tempo_tokens: int = 32
    n_instrument_tokens: int = 17
    n_mood_tokens: int = 16
    
    # Training
    tie_weights: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Initialization
    init_std: float = 0.02


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin cache
        self._update_cache(max_seq_len)
    
    def _update_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._update_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary positional embedding to Q and K."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional Flash Attention."""
    
    def __init__(self, config: MIDITransformerConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.rope = RotaryPositionalEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)
        
        # Handle KV cache for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        if use_cache:
            present_kv = (k, v)
        else:
            present_kv = None
        
        # Apply RoPE
        cos, sin = self.rope(q, k.shape[2])
        q_pos = q.shape[2]
        k_pos = k.shape[2]
        
        # Adjust for cached positions
        cos_q = cos[-q_pos:].unsqueeze(0).unsqueeze(0)
        sin_q = sin[-q_pos:].unsqueeze(0).unsqueeze(0)
        cos_k = cos[:k_pos].unsqueeze(0).unsqueeze(0)
        sin_k = sin[:k_pos].unsqueeze(0).unsqueeze(0)
        
        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k = (k * cos_k) + (rotate_half(k) * sin_k)
        
        # Attention
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use Flash Attention if available
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attn_mask = attn_mask.expand(batch_size, self.n_heads, q.shape[2], k.shape[2])
                attn_mask = attn_mask.bool()
            
            # Causal mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True if past_kv is None else False,
            )
        else:
            # Standard attention
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Causal mask
            if past_kv is None:
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                    diagonal=1
                )
                attn = attn.masked_fill(causal_mask, float("-inf"))
            
            if attention_mask is not None:
                attn = attn.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        # Reshape back
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        
        return out, present_kv


class FeedForward(nn.Module):
    """Feed-forward network with SwiGLU activation."""
    
    def __init__(self, config: MIDITransformerConfig):
        super().__init__()
        hidden_dim = int(config.d_ff * 2 / 3)  # SwiGLU uses 2/3 of standard FFN dim
        
        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down(silu(gate(x)) * up(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        out = self.down_proj(gate * up)
        return self.dropout(out)


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""
    
    def __init__(self, config: MIDITransformerConfig):
        super().__init__()
        self.config = config
        
        self.attn_norm = nn.RMSNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        
        self.ff_norm = nn.RMSNorm(config.d_model)
        self.ff = FeedForward(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        # Pre-norm attention
        residual = x
        x = self.attn_norm(x)
        x, present_kv = self.attn(x, attention_mask, use_cache, past_kv)
        x = residual + x
        
        # Pre-norm FFN
        residual = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = residual + x
        
        return x, present_kv


class MIDITransformer(nn.Module):
    """
    Conditional MIDI Transformer for music generation.
    
    Architecture:
    - Token embeddings with conditioning tokens
    - Rotary positional embeddings (RoPE)
    - Pre-norm transformer blocks with SwiGLU
    - Tied input/output embeddings (optional)
    """
    
    def __init__(self, config: MIDITransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[list] = None,
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            labels: Target token IDs for loss (batch, seq_len)
            use_cache: Whether to return KV cache
            past_key_values: Cached KV for generation
            
        Returns:
            Dict with logits, loss, and optional cache
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.embed_tokens(input_ids)
        x = self.embed_dropout(x)
        
        # Apply transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if self.config.gradient_checkpointing and self.training:
                x, present_kv = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, use_cache, past_kv,
                    use_reentrant=False
                )
            else:
                x, present_kv = layer(x, attention_mask, use_cache, past_kv)
            
            if use_cache:
                present_key_values.append(present_kv)
        
        # Output projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        # NOTE: Dataloaders already offset labels by +1 (labels[i] = next token
        # for input_ids[i]), so no extra shift is needed here.
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": present_key_values,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        Generate MIDI tokens autoregressively.
        
        Args:
            input_ids: Conditioning tokens (batch, prefix_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            
        Returns:
            Generated token sequence (batch, seq_len)
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            if past_key_values is None:
                curr_input = generated
            else:
                curr_input = generated[:, -1:]
            
            outputs = self.forward(
                curr_input,
                use_cache=True,
                past_key_values=past_key_values,
            )
            
            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated[i].tolist()):
                        logits[i, token_id] /= repetition_penalty
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Check for EOS
            if (next_token == eos_token_id).all():
                break
        
        return generated
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params


# Model size configurations
MODEL_CONFIGS = {
    # ~45M params - fast prototyping
    "tiny": MIDITransformerConfig(
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
    ),
    # ~85M params - model_s_raw (raw data learning)
    "small": MIDITransformerConfig(
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
    ),
    # ~125M params - balanced
    "base": MIDITransformerConfig(
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=3072,
    ),
    # ~193M params - model_m_augmented (augmented data)
    "medium": MIDITransformerConfig(
        d_model=1024,
        n_heads=16,
        n_layers=16,
        d_ff=4096,
    ),
    # ~350M params - for 1M+ synthetic files
    "large": MIDITransformerConfig(
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
    ),
    # ~770M params - for 10M+ files
    "xl": MIDITransformerConfig(
        d_model=1536,
        n_heads=24,
        n_layers=32,
        d_ff=6144,
    ),
}


def create_model(
    size: str = "base",
    vocab_size: int = 512,
    **kwargs
) -> MIDITransformer:
    """Create a model with predefined configuration."""
    config = MODEL_CONFIGS.get(size, MODEL_CONFIGS["base"])
    config.vocab_size = vocab_size
    
    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)
    
    return MIDITransformer(config)

