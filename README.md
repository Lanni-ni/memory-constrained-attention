

## Forgetting Attention Kernel (`forgetting_attention_std.py`)

**Input**: Q, K, V, log_fgate (forget gate in log space)

**Output**: Attended values with content-dependent decay

## Stick-breaking Attention (`stickbreaking_attention_std.py`)

**Input**: Q, K, V

**Output**: Attended values with sequential budget allocation

Wrapper around official Triton kernel. Attention budget is allocated sequentially — closer tokens claim first, distant tokens get remainder.
