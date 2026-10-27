# Assignment 1 提交说明

- **姓名**：朱子玥
- **学号**：10230720412

## 正式 W&B Run 链接

### 任务一：BPE

- Run 名称：`10230720412-bpe`
- 链接：https://wandb.ai/jinli0624-east-china-normal-university/lmfs-assignment1/runs/1624ca5fca15
- 主要结果：
  - `bpe/roundtrip_passed`：True
  - `bpe/valid_seconds`：293.48714
  - `bpe/vocab_size`：32000
  - 数据：`/data/assignment1-data/owt_train.txt`、`/data/assignment1-data/owt_valid.txt`

### 任务二：Transformer

- Run 名称：`10230720412-transformer`
- 链接：https://wandb.ai/jinli0624-east-china-normal-university/lmfs-assignment1/runs/66f14910d821
- 主要结果：
  - `val/ppl`：44.9455
  - `train/final_step`：27498
  - `time/total_seconds`：21353.50
  - `time/within_6h`：True
  - 使用任务一分词器：`outputs/1624ca5fca15/tokenizer`

## 续训 / 关联 Run（如需）

本次正式提交的 Transformer 为单次完整训练，无续训累计；`previous_elapsed_seconds = 0`，`resume_from_run = None`。

另有一次实验 run（PPL 更低但总时长略超 6 小时约 24 秒，不作为本次正式提交）：

- https://wandb.ai/jinli0624-east-china-normal-university/lmfs-assignment1/runs/bb4b3fedca96
- `val/ppl`：39.3349
- `time/within_6h`：False

## 说明

1. 两次正式实验均已在 W&B 中记录源码快照、评分所需指标与日志。
2. BPE 可在 Charts 查看 `bpe/first_200_tokens`，在 Summary 查看验证耗时与还原结果。
3. Transformer 可在 Charts 查看 `train/ppl` 曲线，在 Summary 查看最终 `val/ppl` 与计时信息。
