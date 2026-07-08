
```python
prefix = [
    tokens["BOS"],          # 索引 0
    tokens["RANDOM"],       # 索引 1
    tokens["KEY"],          # 索引 2
    *key,                   # 索引 3 ~ 3+q-1
    tokens["SEP"],          # 索引 3+q
]
answer_start = len(prefix)  # = 3 + q
answer_end = answer_start + config.S
input_ids = [*prefix, *target, tokens["EOS"]]
```

## 用默认参数举例（q=4, S=384, V=1024）

```
input_ids:  [ BOS,  RANDOM,  KEY,  k₁,  k₂,  k₃,  k₄,  SEP,  y₁, y₂, ... y₃₈₄, EOS ]
索引:         0      1        2     3     4     5     6     7     8    9       391    392
                              ←── prefix (7 tokens) ──→  ←── target (384 tokens) ──→
                                                            ↑answer_start=8
                                                            └──────────→answer_end=392
```

- **`answer_start = len(prefix) = 3 + q = 7`**（前缀占的 token 数）
- **`answer_end = answer_start + S = 7 + 384 = 391`**

`answer_start` 就是 SEP 后面第一个答案 token 的位置，`answer_end` 是最后一个答案 token 的下一个位置。训练时 `AnswerOnlyCollator` 用这两个数做 label mask（`hjj/phase_c_training.py#L37-L39`）：

```python
labels[row, answer_start - 1 : answer_end - 1] = sequence[answer_start:answer_end]
# 其余位置保持 -100（被 loss 忽略）
```

所以只有 `y₁` 到 `y₃₈₄` 这 384 个 token 产生 loss，前缀和 EOS 不参与。
