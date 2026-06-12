import jax
import jax.numpy as jnp
import flax.linen as nn

class PositionalEncoding(nn.Module):
    embed_dim: int
    max_len: int = 500

    @nn.compact
    def __call__(self, x):
        # x is of shape (batch_size, seq_len, embed_dim)
        seq_len = x.shape[1]
        pe = jnp.zeros((self.max_len, self.embed_dim))
        position = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, jnp.newaxis]
        div_term = jnp.exp(jnp.arange(0, self.embed_dim, 2, dtype=jnp.float32) * -(jnp.log(10000.0) / self.embed_dim))
        
        # Fill even and odd indices
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        # Add to input
        return x + pe[jnp.newaxis, :seq_len, :]

class CausalLinearAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x):
        # x shape: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        out_dim = self.num_heads * self.head_dim
        
        # Project to Q, K, V
        q = nn.Dense(out_dim, name="q_proj")(x)
        k = nn.Dense(out_dim, name="k_proj")(x)
        v = nn.Dense(out_dim, name="v_proj")(x)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        k = k.reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        v = v.reshape((batch_size, seq_len, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        
        # Apply feature map (e.g. ELU + 1) to ensure non-negativity
        phi_q = nn.elu(q) + 1.0
        phi_k = nn.elu(k) + 1.0
        
        # Since it is a causal sequence, we must compute causal linear attention.
        # In a batch causal setup, instead of just multiplying phi_q * (phi_k^T * v),
        # the causal mask requires cumulative sums (since at step t we only attend to steps <= t).
        # We can implement this very cleanly using jnp.cumsum:
        # Numerator: sum_{i=1}^t phi_k_i * v_i^T
        # Denominator: sum_{i=1}^t phi_k_i
        
        # Compute outer product of phi_k and v: shape (batch, heads, seq_len, head_dim, head_dim)
        # We want: outer[b, h, t, i, j] = phi_k[b, h, t, i] * v[b, h, t, j]
        phi_k_expanded = jnp.expand_dims(phi_k, axis=-1)   # (batch, heads, seq_len, head_dim, 1)
        v_expanded = jnp.expand_dims(v, axis=-2)           # (batch, heads, seq_len, 1, head_dim)
        outer_prod = phi_k_expanded * v_expanded           # (batch, heads, seq_len, head_dim, head_dim)
        
        # Cumulative sum over the sequence dimension (axis=2)
        kv_cum = jnp.cumsum(outer_prod, axis=2)            # (batch, heads, seq_len, head_dim, head_dim)
        
        # Multiply phi_q by cumulative kv:
        # result[b, h, t, j] = sum_i phi_q[b, h, t, i] * kv_cum[b, h, t, i, j]
        # Using einops-like jnp.matmul/einsum
        q_expanded = jnp.expand_dims(phi_q, axis=-2)       # (batch, heads, seq_len, 1, head_dim)
        num = jnp.matmul(q_expanded, kv_cum)               # (batch, heads, seq_len, 1, head_dim)
        num = jnp.squeeze(num, axis=-2)                    # (batch, heads, seq_len, head_dim)
        
        # Denominator cumulative sum:
        k_cum = jnp.cumsum(phi_k, axis=2)                  # (batch, heads, seq_len, head_dim)
        den = jnp.sum(phi_q * k_cum, axis=-1, keepdims=True) # (batch, heads, seq_len, 1)
        
        # Avoid division by zero
        out = num / (den + 1e-6)
        
        # Reshape back to (batch, seq_len, embed_dim)
        out = out.transpose((0, 2, 1, 3)).reshape((batch_size, seq_len, out_dim))
        
        # Final output projection
        out = nn.Dense(embed_dim, name="out_proj")(out)
        return out

class LinearTransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        # LayerNorm -> Attention -> Residual
        norm_x = nn.LayerNorm()(x)
        attn_out = CausalLinearAttention(num_heads=self.num_heads, head_dim=self.head_dim)(norm_x)
        x = x + attn_out
        
        # LayerNorm -> MLP -> Residual
        norm_x = nn.LayerNorm()(x)
        mlp_out = nn.Dense(self.mlp_dim)(norm_x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(x.shape[-1])(mlp_out)
        x = x + mlp_out
        
        return x

class ProbabilisticForecaster(nn.Module):
    horizon: int
    num_quantiles: int
    embed_dim: int = 64
    num_heads: int = 4
    head_dim: int = 16
    mlp_dim: int = 128
    num_layers: int = 2

    @nn.compact
    def __call__(self, x):
        # x shape: (batch_size, lookback, num_features)
        
        # Project features to embed_dim
        h = nn.Dense(self.embed_dim)(x)
        
        # Add positional encoding
        h = PositionalEncoding(embed_dim=self.embed_dim)(h)
        
        # Apply Transformer blocks
        for _ in range(self.num_layers):
            h = LinearTransformerBlock(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                mlp_dim=self.mlp_dim
            )(h)
            
        # Global pooling / extraction of final sequence step or flattening
        # Using the last step representation to predict the future is standard in autoregressive settings,
        # but flattening or global average pooling can capture the whole lookback window.
        # Let's pool by flattening or taking the mean/last. Taking the last step + mean works best.
        last_step = h[:, -1, :] # (batch_size, embed_dim)
        mean_steps = jnp.mean(h, axis=1) # (batch_size, embed_dim)
        combined = jnp.concatenate([last_step, mean_steps], axis=-1)
        
        # Map to outputs: shape (batch_size, horizon * num_quantiles)
        out = nn.Dense(128)(combined)
        out = nn.gelu(out)
        out = nn.Dense(self.horizon * self.num_quantiles)(out)
        
        # Reshape to (batch_size, horizon, num_quantiles)
        out = out.reshape((-1, self.horizon, self.num_quantiles))
        
        # To guarantee monotonic quantiles (so higher quantiles are always larger),
        # we can predict the median, and predict exponential diffs for upper/lower quantiles.
        # However, our quantile crossing penalty in losses.py handles this during optimization,
        # which shows a clean mathematical penalty formulation. Let's keep it as raw linear output
        # to let the penalty do the work, showing the optimization power!
        return out
