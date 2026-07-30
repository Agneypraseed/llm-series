# LLM Series

A build-first journey through recurrent sequence models, tokenization, attention,
and Transformers.

## Learning path

```text
    → RNN cell
    → LSTM / GRU
    → Seq2Seq without attention
    → Luong attention
    → Query, Key, and Value
    → Self-attention
    → Multi-head attention
    → Full Transformer
```

## Progress

### First task — Byte Pair Encoding (BPE)

- [ ] Understand why language models tokenize text
- [ ] Build a character-level starting vocabulary
- [ ] Count adjacent token-pair frequencies
- [ ] Select and merge the most frequent pair
- [ ] Repeat merges while updating the vocabulary
- [ ] Implement `train`, `encode`, and `decode`
- [ ] Define behavior for unknown text
- [ ] Test that `decode(encode(text)) == text`
- [ ] Inspect how vocabulary size changes tokenization
- [ ] Compare the implementation with a production tokenizer
- [ ] Document findings and examples

### Phase 1 — Vanilla RNN

- [ ] Understand the RNN recurrence and output projection
- [ ] Explain the roles of input, hidden state, weights, and initial state
- [ ] Implement an RNN cell without `nn.RNN`
- [ ] Unroll the cell over a complete sequence
- [ ] Verify all input, hidden-state, and output shapes
- [ ] Build a synthetic sequence-classification dataset
- [ ] Train a classifier using the final hidden state
- [ ] Explain why the same cell is reused at every time step
- [ ] Inspect vanishing and exploding gradients experimentally
- [ ] Derive the recurrence manually for two or three steps

**Project 1:** Sequence classification

- [ ] Predict a label from a synthetic integer sequence
- [ ] Confirm that the model can overfit 20 examples
- [ ] Trace one example using exact values and tensor shapes

### Phase 2 — LSTM and GRU

- [ ] Understand LSTM cell state and hidden state
- [ ] Understand forget, input, and output gates
- [ ] Understand GRU update and reset gates
- [ ] Implement either an LSTM cell or GRU cell manually
- [ ] Compare the manual cell with `nn.LSTM` or `nn.GRU`
- [ ] Keep input and output shapes identical in the comparison
- [ ] Train vanilla RNN and GRU models on lengths 5, 20, and 50
- [ ] Record loss, accuracy, and gradient norms
- [ ] Explain why gated recurrence handles long-range information better

### Phase 3 — Seq2Seq without attention

- [ ] Understand the encoder–decoder architecture
- [ ] Use the encoder's final state to initialize the decoder
- [ ] Understand `<BOS>` and `<EOS>` tokens
- [ ] Prepare shifted decoder inputs and expected outputs
- [ ] Understand and implement teacher forcing
- [ ] Separate `Encoder`, `Decoder`, and `Seq2Seq`
- [ ] Return all encoder outputs for later attention work
- [ ] Write the greedy autoregressive decoding loop manually
- [ ] Keep training and inference code paths separate

**Project 2:** Toy sequence reversal

- [ ] Reverse sequences of synthetic tokens
- [ ] Train on sequence lengths 3–10
- [ ] Test on lengths 5, 10, 20, and 30
- [ ] Record how performance changes as sequences become longer
- [ ] Explain the fixed-context-vector bottleneck

### Phase 4 — Luong attention

- [ ] Understand alignment scores
- [ ] Implement dot-product attention first
- [ ] Understand general and concat scoring
- [ ] Convert scores to normalized attention weights
- [ ] Compute a weighted context vector
- [ ] Combine context with the decoder state
- [ ] Support source padding masks
- [ ] Test the attention layer independently
- [ ] Assert that weights sum to one
- [ ] Add attention to the Seq2Seq decoder

**Project 3:** Sequence reversal with Luong attention

- [ ] Repeat Project 2 with attention
- [ ] Compare short- and long-sequence performance
- [ ] Save attention weights during inference
- [ ] Visualize attention as a token-alignment heat map
- [ ] Check whether reversal produces a reverse-diagonal pattern

### Phase 5 — Query, Key, and Value

- [ ] Map the decoder state to a query
- [ ] Map encoder states to keys and values
- [ ] Express Luong attention as `softmax(QKᵀ)V`
- [ ] Complete a tiny Q/K/V calculation by hand
- [ ] Reproduce the calculation in PyTorch
- [ ] Explain the role of each tensor without notes

### Phase 6 — Scaled dot-product attention

- [ ] Implement `softmax(QKᵀ / √dₖ)V` from scratch
- [ ] Explain why scores are divided by `√dₖ`
- [ ] Support an optional mask
- [ ] Verify the shapes of Q, K, V, scores, weights, and output
- [ ] Assert that every attention row sums to one
- [ ] Compare the implementation with a library equivalent

### Phase 7 — Self-attention

- [ ] Project the same input into Q, K, and V
- [ ] Implement a single-head `SelfAttention` module
- [ ] Explain self-attention versus cross-attention
- [ ] Inspect how each input token mixes information from other tokens

**Project 4:** Contextual token representations

- [ ] Run self-attention over a sentence containing an ambiguous pronoun
- [ ] Inspect the pronoun's attention weights
- [ ] Explain why attention weights are not guaranteed explanations

### Phase 8 — Causal self-attention

- [ ] Create a lower-triangular causal mask
- [ ] Apply the mask before softmax
- [ ] Assert that future-token weights are exactly zero
- [ ] Explain why causal masking is required
- [ ] Explain how the mask enables parallel training
- [ ] Explain why inference remains autoregressive

### Phase 9 — Multi-head attention

- [ ] Implement Q, K, and V projections
- [ ] Split the model dimension into attention heads
- [ ] Run scaled dot-product attention for every head
- [ ] Concatenate heads
- [ ] Implement the output projection
- [ ] Assert that the model dimension is divisible by the head count
- [ ] Verify that output shape matches input shape
- [ ] Compare the manual implementation with `nn.MultiheadAttention`

### Phase 10 — Full Transformer

- [ ] GPT 2
