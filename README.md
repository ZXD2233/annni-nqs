# ANNNI NQS

An autoregressive neural-network demonstration of ground-state energies and Rényi-2 entanglement entropy in 12/16-spin ANNNI chains, checked against exact diagonalization. Includes training curves and SWAP sampling errors.

## Results

Tested open chains with N = 12, 16 and κ = 0, 0.25, 0.5, 0.75, with the transverse field and nearest-neighbor coupling set to 1. Each point used 3 seeds and 3,600 training steps: 24 runs in total.

| Spins | Maximum energy error per spin | Maximum half-chain S₂ error (bits) |
| --- | --- | --- |
| 12 | 1.00 × 10⁻⁴ | 0.00264 |
| 16 | 1.85 × 10⁻⁴ | 0.00488 |

Errors are absolute differences from exact diagonalization, taking the worst run for each size. SWAP entropy sampling RMSE fell from 0.0186 to 0.00560 bits for N = 12, and from 0.0214 to 0.00539 bits for N = 16, as the number of independent sample pairs increased from 2,000 to 32,000 (10 sampling repeats per trained state). These RMSEs are measured against the enumerated NQS entropy, separating sampling error from wavefunction error.

The result is a small demonstration that validates both energy and entanglement, beyond fitting energy alone.

## Run

Requires Python 3.12. Run from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.5.4 scipy==1.18.1 jax==0.12.0.1 \
  flax==0.12.10 optax==0.2.8 matplotlib==3.11.2 ipykernel==7.4.0
```

Open `ANNNI.ipynb` in VS Code, select this folder's `.venv` Python kernel, and click **Run All**. Edit parameters in the second code cell.

Adapted from [AdaptiveRNNWaveFunctions](https://github.com/jakemcnaughton/AdaptiveRNNWaveFunctions); method: [Recurrent Neural Network Wave Functions](https://arxiv.org/abs/2002.02973). See [LICENSE](LICENSE) for the original upstream terms.

## 中文

用自回归神经网络计算12/16自旋ANNNI链的基态能量与二阶Rényi纠缠熵，并与精确解比较。包含训练曲线和SWAP抽样误差图。

### 结果

完成 N = 12、16，κ = 0、0.25、0.5、0.75 的开边界链计算，横场与最近邻耦合均设为1。每个参数点使用3个随机种子、训练3,600步，共24次训练。

| 自旋数 | 最大每自旋能量误差 | 最大半链 S₂ 误差（bits） |
| --- | --- | --- |
| 12 | 1.00 × 10⁻⁴ | 0.00264 |
| 16 | 1.85 × 10⁻⁴ | 0.00488 |

表中为相对精确对角化结果的绝对误差，取各尺寸全部训练中的最大值。独立样本对从2,000增加到32,000时，SWAP熵估计的RMSE在 N = 12 时从0.0186降至0.00560 bits，在 N = 16 时从0.0214降至0.00539 bits；每个训练态重复采样10次。该RMSE以NQS波函数枚举得到的熵为参照，区分采样误差与波函数误差。

实验结果：除了拟合能量，还验证了纠缠熵，并单独检验了采样误差。

### 运行

需要 Python 3.12。先在项目目录执行上方命令，建立环境并安装全部依赖；依赖已合并到本文件。

用 VS Code 打开 `ANNNI.ipynb`，右上角选择本目录 `.venv` 对应的 Python 内核，然后点击“全部运行”。参数在第二个代码单元格中修改。

代码来源和方法论文见上方链接；许可见 [LICENSE](LICENSE)，其中保留上游英文原版。
