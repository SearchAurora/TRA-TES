"""
tes.py — Trajectory Event Summarizer
=====================================
EUREKA-TRA 的核心创新模块，对轨迹进行三层时序归因分析。

Layer 1 - 时间步归因 (Timestep Attribution):
  滑动窗口计算奖励梯度，检测奖励急剧下降的"崩溃窗口"。
  输出：(crash_start, crash_end, severity, gradient)

Layer 2 - 分量归因 (Component Attribution):
  对每个奖励分量计算其对总奖励方差的贡献度，
  找出导致问题最大的分量。
  输出：[(component_name, contribution_ratio, trend)]

Layer 3 - 阶段归因 (Phase Attribution):
  将 episode 分为 early/mid/late 三段，
  统计各阶段的平均奖励和失败率。
  输出：{phase: {mean_reward, failure_rate, dominant_issue}}

最终将三层归因转化为结构化自然语言报告，注入 LLM prompt。
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════
# Data Classes
# ══════════════════════════════════════════════════════════════════════

@dataclass
class CrashWindow:
    """崩溃窗口：奖励急剧下降的时间段"""
    start_step: int
    end_step: int
    severity: float      # 0-1, 越高越严重
    reward_drop: float   # 奖励下降幅度
    gradient: float      # 平均梯度（负值）


@dataclass
class ComponentAttribution:
    """分量归因结果"""
    name: str
    variance_contribution: float   # 方差贡献比（0-1）
    mean_value: float
    trend: str                     # "increasing" | "decreasing" | "stable"
    is_problematic: bool           # 是否为主要问题分量


@dataclass
class PhaseStats:
    """阶段统计"""
    phase: str                     # "early" | "mid" | "late"
    step_range: Tuple[int, int]
    mean_reward: float
    std_reward: float
    failure_rate: float            # 该阶段 episode 提前结束的比例
    dominant_component: str        # 该阶段贡献最大的奖励分量


@dataclass
class TESReport:
    """TES 完整分析报告"""
    crash_windows: List[CrashWindow]
    component_attributions: List[ComponentAttribution]
    phase_stats: List[PhaseStats]
    summary: str                   # 自然语言总结
    n_episodes: int
    avg_episode_length: float


# ══════════════════════════════════════════════════════════════════════
# Layer 1: Timestep Attribution
# ══════════════════════════════════════════════════════════════════════

def layer1_timestep_attribution(
    episodes: List[Dict],
    window_size: int = 10,
    severity_threshold: float = 0.3,
) -> List[CrashWindow]:
    """
    Layer 1: 滑动窗口检测奖励崩溃时间段。

    算法：
      1. 对所有 episode 的 total_reward 按时间步对齐取均值
      2. 用滑动窗口计算局部梯度
      3. 找到梯度最负（下降最快）的窗口作为崩溃窗口

    Args:
        episodes: 轨迹数据列表
        window_size: 滑动窗口大小
        severity_threshold: 严重度阈值（0-1），低于此不报告

    Returns:
        检测到的崩溃窗口列表，按严重度降序
    """
    if not episodes:
        return []

    # 按时间步对齐（取最短 episode 的长度作为对齐基准）
    max_len = max(ep["length"] for ep in episodes)
    min_len = min(ep["length"] for ep in episodes)
    align_len = min(max_len, 500)  # 最多看 500 步

    # 收集对齐后的奖励曲线
    aligned_rewards = []
    for ep in episodes:
        r = ep["total_reward"]
        if len(r) >= window_size * 2:
            # 如果 episode 比 align_len 短，用最后一个值填充
            padded = np.pad(r, (0, max(0, align_len - len(r))),
                           mode='edge')[:align_len]
            aligned_rewards.append(padded)

    if len(aligned_rewards) < 3:
        return []

    # 平均奖励曲线
    reward_curve = np.mean(aligned_rewards, axis=0)

    # 滑动窗口梯度
    crash_windows = []
    for i in range(0, len(reward_curve) - window_size):
        window = reward_curve[i:i + window_size]
        gradient = (window[-1] - window[0]) / window_size
        reward_drop = window[0] - window[-1]

        if gradient < 0:
            # 归一化严重度
            reward_range = np.max(reward_curve) - np.min(reward_curve)
            if reward_range > 1e-6:
                severity = min(1.0, abs(reward_drop) / reward_range)
            else:
                severity = 0.0

            if severity >= severity_threshold:
                crash_windows.append(CrashWindow(
                    start_step=i,
                    end_step=i + window_size,
                    severity=severity,
                    reward_drop=reward_drop,
                    gradient=gradient,
                ))

    # 合并重叠窗口，只保留最严重的
    crash_windows = _merge_crash_windows(crash_windows)
    crash_windows.sort(key=lambda w: w.severity, reverse=True)

    return crash_windows[:5]  # 最多报告 5 个


def _merge_crash_windows(windows: List[CrashWindow]) -> List[CrashWindow]:
    """合并重叠的崩溃窗口，保留最严重的。"""
    if len(windows) <= 1:
        return windows

    # 按起始步排序
    windows.sort(key=lambda w: w.start_step)
    merged = [windows[0]]

    for w in windows[1:]:
        prev = merged[-1]
        if w.start_step <= prev.end_step:
            # 重叠，保留更严重的
            if w.severity > prev.severity:
                merged[-1] = CrashWindow(
                    start_step=prev.start_step,
                    end_step=max(prev.end_step, w.end_step),
                    severity=w.severity,
                    reward_drop=max(prev.reward_drop, w.reward_drop),
                    gradient=min(prev.gradient, w.gradient),
                )
            else:
                merged[-1] = CrashWindow(
                    start_step=prev.start_step,
                    end_step=max(prev.end_step, w.end_step),
                    severity=prev.severity,
                    reward_drop=max(prev.reward_drop, w.reward_drop),
                    gradient=min(prev.gradient, w.gradient),
                )
        else:
            merged.append(w)

    return merged


# ══════════════════════════════════════════════════════════════════════
# Layer 2: Component Attribution
# ══════════════════════════════════════════════════════════════════════

def layer2_component_attribution(
    episodes: List[Dict],
) -> List[ComponentAttribution]:
    """
    Layer 2: 计算每个奖励分量对总奖励方差的贡献度。

    算法：
      1. 收集所有 episode 的奖励分量时间序列
      2. 计算每个分量的方差
      3. 计算分量间的协方差贡献
      4. 排名并判断趋势

    Returns:
        分量归因列表，按方差贡献降序排列
    """
    if not episodes:
        return []

    # 收集所有分量数据
    all_components = {}
    for ep in episodes:
        for comp_name, comp_values in ep["reward_components"].items():
            if comp_name not in all_components:
                all_components[comp_name] = []
            all_components[comp_name].append(comp_values)

    if not all_components:
        return []

    # 对每个分量：拼接所有 episode 的数据
    comp_stats = {}
    for name, value_lists in all_components.items():
        all_values = np.concatenate(value_lists)
        comp_stats[name] = {
            "variance": np.var(all_values),
            "mean": np.mean(all_values),
            "values": all_values,
        }

    # 计算总方差
    total_variance = sum(s["variance"] for s in comp_stats.values())
    if total_variance < 1e-8:
        total_variance = 1.0

    # 计算趋势（对齐后取前后半段比较）
    def _compute_trend(values: np.ndarray) -> str:
        if len(values) < 20:
            return "stable"
        mid = len(values) // 2
        first_half_mean = np.mean(values[:mid])
        second_half_mean = np.mean(values[mid:])
        change = second_half_mean - first_half_mean
        threshold = 0.1 * (abs(first_half_mean) + 1e-6)
        if change > threshold:
            return "increasing"
        elif change < -threshold:
            return "decreasing"
        return "stable"

    results = []
    for name, stats in comp_stats.items():
        var_contribution = stats["variance"] / total_variance
        trend = _compute_trend(stats["values"])

        # 判断是否为问题分量：高方差 + 均值为负 或 下降趋势
        is_problematic = (
            var_contribution > 0.15 and
            (stats["mean"] < 0 or trend == "decreasing")
        )

        results.append(ComponentAttribution(
            name=name,
            variance_contribution=var_contribution,
            mean_value=stats["mean"],
            trend=trend,
            is_problematic=is_problematic,
        ))

    results.sort(key=lambda c: c.variance_contribution, reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════
# Layer 3: Phase Attribution
# ══════════════════════════════════════════════════════════════════════

def layer3_phase_attribution(
    episodes: List[Dict],
) -> List[PhaseStats]:
    """
    Layer 3: 将 episode 分为 early/mid/late 三段分析。

    算法：
      1. 将每个 episode 按时间步三等分
      2. 统计每段的平均奖励、方差、提前终止率
      3. 找出每段中贡献最大的奖励分量

    Returns:
        三个阶段的统计列表
    """
    if not episodes:
        return []

    phases = ["early", "mid", "late"]
    phase_results = []

    for phase_idx, phase_name in enumerate(phases):
        phase_rewards = []
        phase_early_terminations = 0
        phase_component_sums = {}
        total_episodes = 0

        for ep in episodes:
            ep_len = ep["length"]
            if ep_len < 9:  # 太短无法三等分
                continue

            total_episodes += 1
            third = ep_len // 3
            start = phase_idx * third
            end = (phase_idx + 1) * third if phase_idx < 2 else ep_len

            # 阶段奖励
            phase_r = ep["total_reward"][start:end]
            phase_rewards.extend(phase_r.tolist())

            # 检查是否在该阶段提前终止（episode 长度未达满）
            if phase_idx == 2 and ep_len < 450:  # late phase 提前结束
                phase_early_terminations += 1
            elif phase_idx == 1 and ep_len < 300:  # mid phase 提前结束
                phase_early_terminations += 1
            elif phase_idx == 0 and ep_len < 150:  # early phase 提前结束
                phase_early_terminations += 1

            # 各分量在该阶段的累积值
            for comp_name, comp_values in ep["reward_components"].items():
                segment = comp_values[start:end]
                if comp_name not in phase_component_sums:
                    phase_component_sums[comp_name] = 0.0
                phase_component_sums[comp_name] += np.sum(segment)

        if total_episodes == 0:
            continue

        # 找该阶段的主导分量（绝对值最大）
        dominant = "unknown"
        if phase_component_sums:
            dominant = max(
                phase_component_sums.keys(),
                key=lambda k: abs(phase_component_sums[k])
            )

        phase_rewards_arr = np.array(phase_rewards) if phase_rewards else np.array([0.0])

        phase_results.append(PhaseStats(
            phase=phase_name,
            step_range=(phase_idx * 166, (phase_idx + 1) * 166),  # 近似值
            mean_reward=float(np.mean(phase_rewards_arr)),
            std_reward=float(np.std(phase_rewards_arr)),
            failure_rate=phase_early_terminations / max(total_episodes, 1),
            dominant_component=dominant,
        ))

    return phase_results


# ══════════════════════════════════════════════════════════════════════
# Report Generator
# ══════════════════════════════════════════════════════════════════════

def generate_tes_report(episodes: List[Dict], mode: str = "rules") -> TESReport:
    """
    运行完整的 TES 三层归因分析并生成报告。

    Args:
        episodes: 从 TrajectoryCollector.load() 加载的轨迹数据
        mode: "rules" = v1 固定规则诊断, "raw" = v2 原始数据让 LLM 自分析

    Returns:
        TESReport 对象
    """
    if not episodes:
        return TESReport(
            crash_windows=[],
            component_attributions=[],
            phase_stats=[],
            summary="No trajectory data available for analysis.",
            n_episodes=0,
            avg_episode_length=0.0,
        )

    # 三层分析
    crash_windows = []  # ABL: L1 disabled
    comp_attr = layer2_component_attribution(episodes)
    phase_stats = []  # ABL: L3 disabled

    # 统计
    n_episodes = len(episodes)
    avg_ep_len = np.mean([ep["length"] for ep in episodes])

    # 生成报告（根据 mode 选择格式）
    summary = _build_summary(crash_windows, comp_attr, phase_stats,
                             n_episodes, avg_ep_len, mode=mode)

    return TESReport(
        crash_windows=crash_windows,
        component_attributions=comp_attr,
        phase_stats=phase_stats,
        summary=summary,
        n_episodes=n_episodes,
        avg_episode_length=avg_ep_len,
    )


def _build_summary_rules(
    crash_windows: List[CrashWindow],
    comp_attr: List[ComponentAttribution],
    phase_stats: List[PhaseStats],
    n_episodes: int,
    avg_ep_len: float,
) -> str:
    """[v1] 规则版：用固定启发式规则生成诊断报告。"""
    lines = []

    lines.append(f"=== Trajectory Event Summary ({n_episodes} episodes, avg length {avg_ep_len:.0f} steps) ===\n")

    if crash_windows:
        lines.append("[TIMESTEP ATTRIBUTION]")
        for i, cw in enumerate(crash_windows[:3]):
            lines.append(
                f"  Crash window #{i+1}: steps {cw.start_step}-{cw.end_step}, "
                f"severity={cw.severity:.2f}, reward dropped by {cw.reward_drop:.3f}, "
                f"gradient={cw.gradient:.4f}"
            )
        lines.append("")
    else:
        lines.append("[TIMESTEP ATTRIBUTION] No significant reward crashes detected.\n")

    if comp_attr:
        lines.append("[COMPONENT ATTRIBUTION]")
        for ca in comp_attr:
            flag = " *** PROBLEMATIC ***" if ca.is_problematic else ""
            lines.append(
                f"  {ca.name}: variance_contribution={ca.variance_contribution:.2%}, "
                f"mean={ca.mean_value:.4f}, trend={ca.trend}{flag}"
            )
        problematic = [ca for ca in comp_attr if ca.is_problematic]
        if problematic:
            names = ", ".join(ca.name for ca in problematic)
            lines.append(f"  >> Most problematic components: {names}")
        lines.append("")

    if phase_stats:
        lines.append("[PHASE ATTRIBUTION]")
        for ps in phase_stats:
            lines.append(
                f"  {ps.phase.upper()} phase: "
                f"mean_reward={ps.mean_reward:.4f} (std={ps.std_reward:.4f}), "
                f"failure_rate={ps.failure_rate:.1%}, "
                f"dominant_component={ps.dominant_component}"
            )
        if phase_stats:
            worst_phase = max(phase_stats, key=lambda p: p.failure_rate)
            if worst_phase.failure_rate > 0.1:
                lines.append(
                    f"  >> Worst performing phase: {worst_phase.phase.upper()} "
                    f"(failure_rate={worst_phase.failure_rate:.1%}, "
                    f"dominant issue: {worst_phase.dominant_component})"
                )
        lines.append("")

    lines.append("[SUGGESTED FOCUS]")
    suggestions = []
    if crash_windows:
        cw = crash_windows[0]
        if cw.start_step < 50:
            suggestions.append("Early crash detected — the initial approach reward may be too aggressive or have wrong sign.")
        elif cw.start_step > 300:
            suggestions.append("Late crash detected — the goal-tracking reward may be unstable when the robot is near the target.")
        else:
            suggestions.append(f"Mid-episode crash at steps {cw.start_step}-{cw.end_step} — check if the lifting/transition reward is well-shaped.")
    problematic = [ca for ca in comp_attr if ca.is_problematic]
    if problematic:
        for ca in problematic[:2]:
            if ca.trend == "decreasing":
                suggestions.append(f"'{ca.name}' is decreasing over time — it may need a higher weight or better shaping.")
            elif ca.mean_value < 0:
                suggestions.append(f"'{ca.name}' has negative mean ({ca.mean_value:.4f}) — check if penalty is too harsh.")
    worst_phases = [p for p in phase_stats if p.failure_rate > 0.2]
    for wp in worst_phases:
        suggestions.append(f"{wp.phase.upper()} phase has {wp.failure_rate:.0%} failure rate — strengthen {wp.dominant_component} reward in this phase.")
    if not suggestions:
        suggestions.append("No critical issues detected. Consider fine-tuning weights for marginal improvement.")
    for s in suggestions:
        lines.append(f"  - {s}")

    return "\n".join(lines)


def _build_summary_raw(
    crash_windows: List[CrashWindow],
    comp_attr: List[ComponentAttribution],
    phase_stats: List[PhaseStats],
    n_episodes: int,
    avg_ep_len: float,
) -> str:
    """
    [v2] 原始数据版：只提供结构化数据，不做诊断判断。
    让 LLM 自己从数据中分析问题所在。
    """
    lines = []

    lines.append(f"=== Raw Trajectory Data ({n_episodes} episodes, avg {avg_ep_len:.0f} steps/episode) ===")
    lines.append(f"Episode timeout: 500 steps. Task: lift cube to target position.\n")

    # 原始崩溃窗口数据
    lines.append("[REWARD GRADIENT DATA]")
    if crash_windows:
        lines.append("Detected reward drop windows (sorted by magnitude):")
        for i, cw in enumerate(crash_windows[:5]):
            lines.append(
                f"  Window {i+1}: steps {cw.start_step}-{cw.end_step}, "
                f"reward_change={-cw.reward_drop:.4f}, "
                f"avg_gradient={cw.gradient:.6f}/step"
            )
    else:
        lines.append("  No significant reward gradient changes detected across timesteps.")
    lines.append("")

    # 原始分量统计 — 完整数据表格
    lines.append("[PER-COMPONENT STATISTICS]")
    if comp_attr:
        lines.append(f"{'Component':<20} {'Mean':>10} {'Std':>10} {'Var%':>8} {'Trend':>12}")
        lines.append("-" * 62)
        for ca in comp_attr:
            std = (ca.variance_contribution * ca.mean_value) if ca.mean_value != 0 else 0
            lines.append(
                f"{ca.name:<20} {ca.mean_value:>10.4f} "
                f"{ca.variance_contribution:>10.2%} "
                f"{ca.variance_contribution:>7.1%} "
                f"{ca.trend:>12}"
            )
        lines.append("")
        # 额外提供分量之间的比例关系
        total_positive = sum(ca.mean_value for ca in comp_attr if ca.mean_value > 0)
        total_negative = sum(ca.mean_value for ca in comp_attr if ca.mean_value < 0)
        lines.append(f"  Total positive reward per step: {total_positive:.4f}")
        lines.append(f"  Total negative reward per step: {total_negative:.4f}")
        lines.append(f"  Net reward per step: {total_positive + total_negative:.4f}")
    lines.append("")

    # 原始阶段数据 — 不做判断
    lines.append("[PER-PHASE BREAKDOWN]")
    lines.append("Each episode is divided into three equal phases:")
    if phase_stats:
        lines.append(f"{'Phase':<8} {'Steps':>12} {'Mean R':>10} {'Std R':>10} {'Timeout%':>10} {'Top Component':>18}")
        lines.append("-" * 70)
        for ps in phase_stats:
            lines.append(
                f"{ps.phase.upper():<8} "
                f"{ps.step_range[0]:>4}-{ps.step_range[1]:<6} "
                f"{ps.mean_reward:>10.4f} "
                f"{ps.std_reward:>10.4f} "
                f"{ps.failure_rate:>9.0%} "
                f"{ps.dominant_component:>18}"
            )
    lines.append("")

    # 关键提示：让 LLM 自己分析
    lines.append("[YOUR TASK]")
    lines.append("Analyze the data above and identify:")
    lines.append("  1. Which reward component is the biggest bottleneck and why?")
    lines.append("  2. At what episode phase does performance degrade most?")
    lines.append("  3. What specific structural change to the reward function would address this?")
    lines.append("Do NOT just scale weights. Change the reward STRUCTURE based on your analysis.")

    return "\n".join(lines)


def _build_summary(
    crash_windows: List[CrashWindow],
    comp_attr: List[ComponentAttribution],
    phase_stats: List[PhaseStats],
    n_episodes: int,
    avg_ep_len: float,
    mode: str = "rules",
) -> str:
    """
    生成 TES 报告。

    Args:
        mode: "rules" = v1 固定规则诊断, "raw" = v2 原始数据让 LLM 自分析
    """
    if mode == "rules":
        return _build_summary_rules(crash_windows, comp_attr, phase_stats,
                                     n_episodes, avg_ep_len)
    else:
        return _build_summary_raw(crash_windows, comp_attr, phase_stats,
                                   n_episodes, avg_ep_len)


def report_to_prompt_text(report: TESReport) -> str:
    """
    将 TES 报告转换为可直接注入 LLM prompt 的文本。
    这是 EUREKA-TRA 和 EUREKA 的关键区别。
    """
    return report.summary
