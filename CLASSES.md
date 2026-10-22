# CLIP From Scratch: Class Documentation

Notation: `B` = batch size, `L` = text sequence length (`max_seq_length`), `W` = model width, `E` = joint embedding dim (`emb_dim`).

Dependency order: `PositionalEmbedding` → `AttentionHead` → `MultiHeadAttention` → `TransformerEncoder` → `TextEncoder` / `ImageEncoder` → `CLIP`. `FashionMNIST` is the dataset used for training.

---

## PositionalEmbedding(width, max_seq_length)

Fixed (non-learned) sinusoidal positional encoding, as in "Attention Is All You Need".

- **Args**
  - `width`: embedding dimension `W`.
  - `max_seq_length`: number of positions to precompute.
- **Construction**: builds a `[max_seq_length, width]` table where even dimensions use `sin(pos / 10000^(i/width))` and odd dimensions use `cos(pos / 10000^((i-1)/width))`. The table is stored with `register_buffer('pe', ...)` as shape `[1, L, W]`, so it moves with the module across devices and is saved in the state dict, but it is not trained.
- **forward(x)**: `x` is `[B, L, W]`. Returns `x + pe`, same shape. The sequence length must equal `max_seq_length`.

---

## AttentionHead(width, head_size)

A single head of scaled dot-product self-attention.

- **Args**
  - `width`: input dimension `W`.
  - `head_size`: dimension of this head's Q/K/V projections.
- **Layers**: three `nn.Linear(width, head_size)` for `query`, `key`, `value`.
- **forward(x, mask=None)**
  - `x`: `[B, L, W]`.
  - `mask`: optional tensor broadcastable to `[B, L, L]`. Positions where `mask == 0` are set to `-inf` before the softmax, so they get zero attention weight.
  - Computes `softmax(Q Kᵀ / sqrt(head_size)) V`.
  - Returns `[B, L, head_size]`.

---

## MultiHeadAttention(width, n_heads)

Runs `n_heads` attention heads in parallel and mixes them with an output projection.

- **Args**
  - `width`: model width `W`.
  - `n_heads`: number of heads. `head_size = width // n_heads`, so `width` should be divisible by `n_heads`.
- **Layers**: `heads` (`ModuleList` of `AttentionHead`) and `W_o` (`nn.Linear(width, width)`).
- **forward(x, mask=None)**: runs every head on `x` with the same `mask`, concatenates their outputs along the last dimension, and applies `W_o`. Input and output are both `[B, L, W]`.

---

## TransformerEncoder(width, n_heads, r_mlp=4)

One pre-LayerNorm Transformer encoder block.

- **Args**
  - `width`: model width `W`.
  - `n_heads`: attention heads.
  - `r_mlp`: hidden-layer expansion ratio of the MLP (default 4).
- **Layers**: `ln1`, `mha` (`MultiHeadAttention`), `ln2`, and `mlp` (`Linear(W, W*r_mlp)` → `GELU` → `Linear(W*r_mlp, W)`).
- **forward(x, mask=None)**
  1. `x = x + mha(ln1(x), mask)`
  2. `x = x + mlp(ln2(x))`

  Both sub-layers use residual connections. Shape is unchanged: `[B, L, W]`.

---

## TextEncoder(vocab_size, width, max_seq_length, n_heads, n_layers, emb_dim)

Transformer text tower. It turns tokenized text into one L2-normalized embedding per caption.

- **Args**
  - `vocab_size`: number of token ids (256, since the tokenizer is byte-level).
  - `width`: model width.
  - `max_seq_length`: fixed token length.
  - `n_heads`, `n_layers`: attention heads per block and number of blocks.
  - `emb_dim`: size of the joint image/text embedding space.
- **Layers and parameters**: `encoder_embedding` (`nn.Embedding`), `positional_embedding`, `encoder` (`ModuleList` of `TransformerEncoder`), and `projection` (learned `[width, emb_dim]` matrix).
- **forward(text, mask=None)**
  - `text`: `[B, L]` integer token ids.
  - `mask`: `[B, L, L]` tensor (1 = real token, 0 = padding). It is required: it is passed to the attention layers and is also used to find the EOT position.
  - Steps: embed tokens, add positional encoding, run all encoder blocks, then select the feature at the EOT token (index `sum(mask[:, 0]) - 1`, i.e. the last non-padding token). Project to `emb_dim` and L2-normalize.
  - Returns `[B, emb_dim]` unit vectors.

---

## ImageEncoder(width, img_size, patch_size, n_channels, n_layers, n_heads, emb_dim)

Vision Transformer (ViT) image tower.

- **Args**
  - `width`: model width.
  - `img_size`: `(H, W)` of the input image.
  - `patch_size`: `(pH, pW)`. It must divide `img_size` evenly, and `width` must be divisible by `n_heads`. Both conditions are checked with `assert`.
  - `n_channels`: input channels (1 for grayscale).
  - `n_layers`, `n_heads`: encoder depth and heads per block.
  - `emb_dim`: joint embedding dimension.
- **Attributes**: `n_patches = (H*W)/(pH*pW)` and `max_seq_length = n_patches + 1` (the extra slot is the class token).
- **Layers and parameters**: `linear_project` (a strided `Conv2d` that cuts the image into patches and projects each to `width`), `cls_token` (learned `[1, 1, width]`), `positional_embedding`, `encoder` (`ModuleList` of `TransformerEncoder`), and `projection` (learned `[width, emb_dim]`).
- **forward(x)**
  - `x`: `[B, C, H, W]` image batch.
  - Steps: patchify and flatten to `[B, n_patches, width]`, prepend the class token, add positional encoding, run the encoder blocks (no mask), take the class-token output, project, and L2-normalize.
  - Returns `[B, emb_dim]` unit vectors.

---

## CLIP(emb_dim, vit_width, img_size, patch_size, n_channels, vit_layers, vit_heads, vocab_size, text_width, max_seq_length, text_heads, text_layers)

Contrastive Language-Image Pre-training model. It joins the two encoders and computes the contrastive loss.

- **Args**: the image-tower arguments (`vit_width`, `img_size`, `patch_size`, `n_channels`, `vit_layers`, `vit_heads`), the text-tower arguments (`vocab_size`, `text_width`, `max_seq_length`, `text_heads`, `text_layers`), and the shared `emb_dim`.
- **Attributes**
  - `image_encoder` (`ImageEncoder`) and `text_encoder` (`TextEncoder`).
  - `temperature`: learnable scalar, initialised to `log(1/0.07)`. It is exponentiated in `forward` to scale the logits.
  - `device`: `cuda` if available, otherwise `cpu`. It is used to place the label tensor.
- **forward(image, text, mask=None)**
  1. Encodes both modalities into `I_e` and `T_e`, each `[B, emb_dim]` and normalized.
  2. Computes `logits = (I_e @ T_eᵀ) * exp(temperature)`, a `[B, B]` matrix of scaled cosine similarities.
  3. Uses `labels = arange(B)`, since the matching pair of each sample is on the diagonal.
  4. Computes cross-entropy over image→text and text→image directions and averages them (symmetric InfoNCE loss).
  - Returns a scalar loss tensor. It does not return embeddings or logits. For inference, call `model.image_encoder` and `model.text_encoder` directly, as the final notebook cell does.

---

## FashionMNIST(train=True)  (`torch.utils.data.Dataset`)

Wraps the Hugging Face `zalando-datasets/fashion_mnist` dataset as image/caption pairs for CLIP training.

- **Args**: `train`: `True` selects the `train` split, `False` the `test` split.
- **Attributes**: `dataset` (the loaded DatasetDict), `transform` (`ToTensor`), `split`, and `captions` (a map from the 10 class labels to captions such as `"An image of a t-shirt/top"`).
- **`__len__`**: number of rows in the selected split.
- **`__getitem__(i)`**: returns a dict:
  - `"image"`: float tensor `[1, 28, 28]` in `[0, 1]`.
  - `"caption"`: `[max_seq_length]` int32 token tensor from `tokenizer` (SOT, text bytes, EOT, zero padding).
  - `"mask"`: `[L, L]` int32 tensor, the 1D padding mask repeated on each row. `DataLoader` batches it to `[B, L, L]`, which is the shape `TextEncoder` expects.

---

## Related function: tokenizer(text, encode=True, mask=None, max_seq_length=32)

Not a class, but the classes above depend on it.

- **encode=True**: wraps `text` with `chr(2)` (SOT) and `chr(3)` (EOT), pads with `chr(0)` to `max_seq_length`, and UTF-8 encodes it into an `IntTensor`. Returns `(tokens, mask)`, where `mask` is 1 for real tokens and 0 for padding.
- **encode=False**: decodes `text` back to a string, dropping the SOT and EOT tokens using the mask length. Returns `(string, None)`.
- Limitation: a caption longer than `max_seq_length - 2` bytes would break the padding step, because the negative padding count gives an empty pad and the sequence is not truncated.
