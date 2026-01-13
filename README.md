

## Forgetting Attention Kernel (`forgetting_attention_std.py`)

**Input**: Q, K, V, log_fgate (forget gate in log space)

**Output**: Attended values with content-dependent decay
### Key computation:
```
log_lambda = torch.cumsum(log_fgate, dim=-1)
decay_bias = log_lambda[:,:,:T_q,None] - log_lambda[:,:,None,:]
scores = scores + decay_bias
```
This adds a content-dependent decay bias to attention scores. Unlike ALiBi (distance-only), the decay here is learned from content.

## Stick-breaking Attention (`stickbreaking_attention_std.py`)

**Input**: Q, K, V

**Output**: Attended values with sequential budget allocation

Wrapper around official Triton kernel. Attention budget is allocated sequentially — closer tokens claim first, distant tokens get remainder.
