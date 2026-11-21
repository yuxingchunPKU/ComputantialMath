# 用 Winograd 内积算法加速神经网络线性层

本文把 [Winograd 内积恒等式](../README.md) 推广到全连接层

```
Y = X * W^T + b
```

其中（行主序，与 PyTorch `nn.Linear` 的权重布局一致）：

| 符号 | 形状 | 含义 |
| --- | --- | --- |
| `X` | `m × n` | 一个 batch 的输入，`m` 为 batch size，`n` 为 in_features |
| `W` | `p × n` | 权重，`p` 为 out_features，`n` 为 in_features |
| `b` | `p` | 偏置 |
| `Y` | `m × p` | 输出 |

代码位于 `include/linear.hpp`、`src/linear.cpp`，基准测试在 `src/bench.cpp`
的 `benchmark_linear`。

## 1. 朴素算法

```
for i in 0..m-1:
    for j in 0..p-1:
        s = b[j]
        for k in 0..n-1:
            s += X[i,k] * W[j,k]
        Y[i,j] = s
```

每个输出元素做 `n` 次乘法、`n` 次加法（含偏置），总计

```
乘法 m·p·n，加法 m·p·n
```

## 2. Winograd 恒等式

对一对元素：

```
x0*w0 + x1*w1 = (x0 + w1) * (x1 + w0) - x0*x1 - w1*w0
```

把长度 `n = 2h + r`（`r = n mod 2`）的内积两两分组：

```
x·w = Σ_{k=0}^{h-1} (x_{2k} + w_{2k+1})(x_{2k+1} + w_{2k})
      - Σ_k x_{2k} x_{2k+1}
      - Σ_k w_{2k} w_{2k+1}
      + r · x_{n-1} w_{n-1}
```

定义两个自乘积（self-product）：

```
xs(x) = Σ_k x_{2k} x_{2k+1}
ws(w) = Σ_k w_{2k} w_{2k+1}
```

## 3. 推广到线性层

对 `Y[i,j] = X[i,:] · W[j,:] + b[j]` 逐项套用：

```
Y[i,j] = Σ_k (X[i,2k] + W[j,2k+1]) (X[i,2k+1] + W[j,2k])
         - xs_i - ws_j + b[j]
         + r · X[i,n-1] W[j,n-1]
```

把偏置折叠进权重项，令

```
wterm_j = ws_j - b[j]
```

则

```
Y[i,j] = Σ_k (X[i,2k] + W[j,2k+1]) (X[i,2k+1] + W[j,2k])
         - xs_i - wterm_j
         + r · X[i,n-1] W[j,n-1]
```

这样偏置不需要额外的加法：它和 `ws_j` 一起在预处理阶段算好。

注意 `X[i,2k]`、`W[j,2k]` 都是按行连续存储，主循环是对两行做线性扫描，
没有一般矩阵乘法里 `B` 的列跨步问题，cache 行为更好。

## 4. 算法步骤

```
# 第一遍：每个输入行一个自乘积
for i in 0..m-1:
    xs[i] = Σ_k X[i,2k] * X[i,2k+1]

# 第二遍：每个输出行一个自乘积，并折叠偏置
for j in 0..p-1:
    wterm[j] = Σ_k W[j,2k] * W[j,2k+1] - b[j]

# 第三遍：主循环，每个元素 n/2 次乘法
for i in 0..m-1:
    for j in 0..p-1:
        s = 0
        for k in 0..h-1:
            s += (X[i,2k] + W[j,2k+1]) * (X[i,2k+1] + W[j,2k])
        s -= xs[i] + wterm[j]
        if n is odd:
            s += X[i,n-1] * W[j,n-1]
        Y[i,j] = s
```

对应三个 API：

| 函数 | 用途 |
| --- | --- |
| `linear_naive` | 参考实现 |
| `linear_weight_terms` | 计算 `wterm`，可缓存 |
| `linear_winograd_cached` | 主计算，`wterm` 由调用方提供 |
| `linear_winograd` | 便捷版：每次调用内部重算 `wterm` |

## 5. 运算量

设 `n = 2h` 为偶数（奇数时再加 `m·p` 次乘法和加法）：

| 算法 | 乘法 | 加法 |
| --- | ---: | ---: |
| 朴素 | `m p n` | `m p n` |
| Winograd（`wterm` 缓存） | `m h + m p h` | `m(h-1) + m p (3h+1)` |
| Winograd（每次重算 `wterm`） | `(m+p)h + m p h` | `(m+p)(h-1) + m p (3h+1)` |

乘法数量的比值：

```
Winograd / naive = 1/2 + (m + p) / (2 m p)
```

- `m`、`p` 都较大时趋近 **1/2**；
- `m = 1`（单样本推理）时趋近 **1**，没有乘法收益——因为 `X` 和 `W` 的
  自乘积各要 `n/2` 次乘法，正好抵消；
- 所以这个算法需要**成批**（batch）计算才划算。

加法增加，总浮点运算量仍约为 `2 m p n`，和朴素算法相同。

## 6. 预计算与缓存策略

| 场景 | `wterm` 处理 |
| --- | --- |
| 推理，权重固定 | 每个线性层算一次并缓存，之后只跑 `linear_winograd_cached` |
| 训练，权重每步更新 | 每次 forward 重算 `wterm`（`linear_winograd`）；开销为 `O(p n)` |
| 权重固定、batch 很大 | 重算开销相对 `O(m p n)` 可忽略，缓存与否差别很小 |

`wterm` 只需 `p` 个 double，内存开销可以忽略。输入侧的 `xs` 每行一个，
随 `X` 一起处理，不需要跨调用保存。

## 7. 为什么实际会更快

虽然总运算量几乎不变，单个内积的**依赖链长度**减半：朴素内层要做 `n` 次
`s += x[k]*w[k]`，每次加法都依赖上一次；Winograd 内层只有 `h = n/2` 次
`s += (…)*(…)`。成对求和的两次加法是独立的，可以提前计算，不在累加关键
路径上。因此在标量、延迟受限的代码里，墙钟时间接近减半。

这也是实测中线性层比一般矩阵乘法收益更明显的原因：线性层的主循环访问的是
两行连续内存，而一般 `C = A*B` 的列访问有跨步，额外加法带来的内存压力会
吃掉一部分收益。

如果编译器启用了 FMA、自动向量化或使用了 BLAS，乘法与加法吞吐相同，这一
优势会缩小甚至消失。本项目的对比是算法层面的教学实现，不代表生产性能。

## 8. 实测结果

本机 AMD Ryzen 9 9950X，g++ 11.4，`-O2`，随机 `[-1,1]` 数据（完整输出见
`results.txt`）：

| m | n | p | 算法 | 乘法 | 时间 (ms) |
| ---: | ---: | ---: | --- | ---: | ---: |
| 64 | 256 | 256 | 朴素 | 4,194,304 | 1.147 |
| 64 | 256 | 256 | Winograd | 2,138,112 | 0.755 |
| 128 | 512 | 512 | 朴素 | 33,554,432 | 10.540 |
| 128 | 512 | 512 | Winograd | 16,941,056 | 6.357 |
| 128 | 784 | 256 | 朴素 | 25,690,112 | 8.396 |
| 128 | 784 | 256 | Winograd | 12,995,584 | 4.803 |
| 256 | 1024 | 1024 | 朴素 | 268,435,456 | 89.530 |
| 256 | 1024 | 1024 | Winograd | 134,873,088 | 49.757 |

乘法约减半，实际速度约 1.5–1.8 倍；`wterm` 缓存版只比每次重算快约 1%，
因为 `p·n/2` 相对 `m·p·n/2` 很小。最大绝对误差约 `1e-13`。

## 9. 数值稳定性

公式包含相消项：`(x0+w1)(x1+w0)` 与 `x0x1 + w1w0` 可能都远大于最终结果。
随机数据下误差在 `1e-13` 量级；若输入或权重动态范围很大、或需要高精度累加，
应改用补偿求和（Kahan/Neumaier）或直接使用 BLAS。神经网络推理通常使用
float32 且对 `1e-6` 级误差不敏感，但本实现用 double 演示。

## 10. 使用方法

```cpp
#include "linear.hpp"

// X: m*n, W: p*n, b: p, Y: m*p（均为 row-major）
std::vector<double> wterm(p);
winograd::linear_weight_terms(W.data(), b.data(), wterm.data(), p, n);

// 推理：wterm 缓存后反复调用
winograd::linear_winograd_cached(X.data(), W.data(), wterm.data(), Y.data(),
                                 m, n, p);

// 训练：权重变化时直接用便捷版
winograd::linear_winograd(X.data(), W.data(), b.data(), Y.data(), m, n, p);
```

构建与测试（与主项目共用）：

```bash
cd Winograd
make
./bench --quick
./bench --fc-shapes 256:1024:1024,128:784:256
```

## 11. 与深度学习框架的关系

- PyTorch / TensorFlow 的 `Linear` 最终调用 BLAS 的 `sgemm/dgemm`。经过分块、
  向量化和多线程的 GEMM 远快于本文的朴素循环和 Winograd 循环；
- 卷积网络里的 "Winograd 算法"（`F(2,3)`、`F(4,3)` 等）是另一类极小滤波
  算法，通过多项式变换减少卷积乘法，和本文的内积恒等式不是同一个东西；
- 本文的价值在于展示"减少乘法次数"与"降低实际延迟"不是一回事：现代 CPU
  上加法与乘法同价，Winograd 的收益主要来自更短的关键路径和更少的循环
  迭代，而不是乘法计数。

## 12. 可扩展方向

- 用 OpenMP 并行化 `i` 或 `j` 循环；
- 用 AVX2/AVX-512 手动向量化主循环（成对数据天然连续，适合向量化）；
- 把 `wterm` 与量化/混合精度结合（需注意相消放大误差）；
- 对卷积层先 `im2col` 成 GEMM，再套用本算法，与卷积 Winograd 做对比。
