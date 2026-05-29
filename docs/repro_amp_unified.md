# 复现计划：State-Dependent AMP 统一「走/跑/摔倒恢复」(G1 23dof)

> 交接文档。本文件用于把另一项目里的讨论迁移到 TienKung-Lab 继续。
> 在 TienKung-Lab 目录开新的 Claude Code 会话后，让它先读本文件即可接续。
> 论文：*Unified Walking, Running, and Recovery for Humanoids via State-Dependent
> Adversarial Motion Priors*（HKU，Peng Lu 组，arXiv 2605.18611v1，**无开源代码**）。

## 0. 已拍板的决策
- **基础代码库**：本项目 TienKung-Lab（已有完整 Peng 式 AMP + G1 env）。
- **机器人**：Unitree **G1 23dof**（用户有 23dof 实机；论文用 29dof，我们刻意偏离以匹配实机）。
- **数据**：用户自行从开源数据（LAFAN1 或等价）重定向，产出三段 `.txt`（格式见 §2）。
- **并行策略**：方法实现不依赖新数据，可先用现有 walk 数据做冒烟测试。

## 1. 论文方法要点（必须复现的，不是普通 AMP）
- PPO + AMP 风格奖励，`λ_amp = 0.5`。
- **核心贡献 = 状态门控的双判别器**：
  - 门控函数按投影重力：`|g_z + 1| > 0.6`（躯干倾斜≈37°）→ **recovery 判别器**；否则 → **locomotion 判别器**。
  - 门控**同时路由**：(a) 该 transition 用哪个判别器算 AMP 奖励；(b) 该 transition 喂哪个判别器去训练。
- **recovery 判别器**：只用 `fallAndGetUp` 片段训练，**不带速度条件**。
- **locomotion 判别器**：用 walk + run 联合训练，**速度条件化** —— 归一化速度 `v̂` 作为判别器额外输入；低 `v̂` 采 walk、高 `v̂` 采 run。
- 速度指令范围：常规 −0.5~1.0 m/s，快速模式 −1.5~3.0 m/s。
- 策略：3 层 MLP，观测 96×4=384 维堆叠（angvel, proj_gravity, cmd_vel, 关节相对位置, 关节速度, 上一动作）。动作：目标关节位置，PD 跟踪，50Hz，单 ONNX 部署。

### 论文未写明、需我们自定的参数（动手时记录在此）
- 5 项 task reward 权重 `wv,ws,wp,we,wf`（论文只给符号，仅 `λ_amp=0.5`）。
- 判别器网络层宽（论文只描述策略网络）。
- 判别器观测的确切子集（采用 TienKung 现有 58 维，见 §2）。
- `v̂` 的归一化定义（拟用 `v_cmd / v_max`）。

## 2. 数据规格（硬约束，三段格式完全一致）
论文用三段**独立**片段（**互不连贯**）：`walk1_subject1`、`run1_subject2`、`fallAndGetUp2_subject2`。
- AMP 判别器观测 = **58 维** = joint_pos(23) + joint_vel(23) + end_effector_pos(12: 左手/右手/左脚/右脚相对 root、root 系下)。关节顺序见 `g1_env.get_amp_obs_for_expert_trans`（左腿/右腿/腰/左臂/右臂）。
- 文件 = JSON `.txt`，字段 `LoopMode/FrameDuration/MotionWeight/Frames`；每帧 **65 维** = `root_pos(3)+root_orn(4)+上述58`，`AMPLoader` 自动丢前 7 维。参考现有 `legged_lab/envs/g1/datasets/motion_amp_expert/g1_qie_walk_amp.txt`（FrameDuration=0.033≈30Hz）。
- 需产出三个文件：`g1_walk_amp.txt`、`g1_run_amp.txt`、`g1_fall_recovery_amp.txt`。
- walk/run 各需标注一个归一化速度标签 `v̂`（喂 locomotion 判别器用）；fall 段不需要。
- **每段内部连续即可**（如 fall 段是「躺→起身→站」连续序列），段与段之间不需要拼接——串联由 RL+门控学出。

## 3. 关键代码位置（集成点）
- `rsl_rl/rsl_rl/modules/discriminator.py` — `Discriminator`（trunk+linear, grad_pen, `predict_amp_reward`=coef·clamp(1−¼(d−1)²,0)）。
- `rsl_rl/rsl_rl/algorithms/amp_ppo.py` — `AMPPPO`：
  - `__init__(discriminator, amp_data, ...)` 单判别器 + `amp_storage=ReplayBuffer(input_dim//2,...)`（line ~115-119）。
  - `update()`（line ~224）：line ~253 `amp_expert_generator`，line ~431-436 计算 `policy_d/expert_d` 与 `grad_pen`。
- `rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` — 编排 act/step/奖励融合。
- `rsl_rl/rsl_rl/utils/motion_loader.py` — `AMPLoader`（JOINT_POS=23, JOINT_VEL=23, END_EFF=12, `feed_forward_generator`）。
- `legged_lab/envs/g1/g1_env.py` — `get_amp_obs_for_expert_trans`（line ~567）、proj_gravity、reset。
- `legged_lab/envs/g1/qie_walk_cfg.py` — env/agent 配置（`UNITREE_G1_23DOF_CFG`）。
- `legged_lab/envs/g1/datasets/motion_amp_expert/` — 数据目录。

## 4. 改造点（单→双判别器+门控+速度条件）
1. **门控**：在 env 暴露 `g(s)=|proj_gravity_z+1|>0.6` 的 mask（per-env）。
2. **双判别器**：`AMPPPO` 改为持有 `disc_loco` + `disc_rec`，各自 replay buffer。
   - `disc_loco.input_dim` 需 +1（拼接 `v̂`）；`disc_rec` 维持 58×2。
3. **数据**：`AMPLoader` 支持分组加载（loco: walk+run 带 `v̂`；rec: fall）。loco 采样按当前 `v̂` 在 walk/run 间插值/选择。
4. **奖励路由**：`update()`/runner 里按门控 mask 决定每个 transition 用哪个判别器算 style reward、喂哪个 buffer。
5. **fall 恢复环境**：新增 prone/supine 初始化事件、fall 惩罚、终止逻辑调整、快速模式速度范围。
6. **smoke test**：先只接 loco 判别器（用现有 walk），确认改造不破坏原训练，再加 rec 与门控。

## 5. 分阶段计划（每阶段可独立验证）
- **阶段0**：跑通现有 G1 walk-AMP（单判别器）→ 验证 pipeline、AMP reward 正常。
- **阶段1**：加 run + 速度条件化 loco 判别器 → 验证单策略在 walk/run 速度区间切换。
- **阶段2**：加 fall 数据 + prone/supine 初始化 + rec 判别器 + 投影重力门控 → 验证躺/趴姿能起身转行走。
  - 子验证：打日志确认躺姿走 rec、站姿走 loco 后再整体训练。
- **阶段3**：调 reward 权重等收尾。

## 6. 已知差异/风险
- 23dof vs 论文 29dof（动作维度、AMP obs 维度均不同）。
- 论文超参缺失 → 复现不出逐一对应数字，需自行调参。
- 论文那三段 G1 重定向产物未发布 → 我们自重定向，等价非同一份；动手前先在 LAFAN1 官方确认含 `fallAndGetUp` 片段。
- TienKung 现有 G1 AMP 仅 walk、单判别器，无 run/recovery。
