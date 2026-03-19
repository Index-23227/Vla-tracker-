// Centralized benchmark definitions for the VLA-Tracker dashboard.
// All components should import from here instead of defining their own lists.

/**
 * Core benchmark metadata: avgKey, label, max scale, suites, and metric.
 * This is the single source of truth for benchmark configuration.
 */
export const BENCHMARKS = {
  libero: {
    key: 'libero_avg',
    label: 'LIBERO',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long'],
    suiteLabels: { libero_spatial: 'Spatial', libero_object: 'Object', libero_goal: 'Goal', libero_long: 'Long' },
  },
  calvin: {
    key: 'calvin_avg',
    label: 'CALVIN',
    max: 5,
    metric: 'Avg chain length (max 5)',
    suites: ['calvin_abc_d_avg_len'],
    suiteLabels: { calvin_abc_d_avg_len: 'ABC→D Avg Len' },
  },
  simpler_env: {
    key: 'simpler_avg',
    label: 'SimplerEnv',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['google_robot_pick_coke_can', 'google_robot_move_near'],
    suiteLabels: { google_robot_pick_coke_can: 'Pick Can', google_robot_move_near: 'Move Near' },
  },
  robotwin_v1: {
    key: 'robotwin_v1_avg',
    label: 'RoboTwin v1',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['robotwin_easy', 'robotwin_hard'],
    suiteLabels: { robotwin_easy: 'Easy', robotwin_hard: 'Hard (Randomized)' },
  },
  robotwin_v2: {
    key: 'robotwin_v2_avg',
    label: 'RoboTwin v2',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['short_horizon', 'medium_horizon', 'long_horizon'],
    suiteLabels: { short_horizon: 'Short', medium_horizon: 'Medium', long_horizon: 'Long' },
  },
  rlbench: {
    key: 'rlbench_avg',
    label: 'RLBench',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['rlbench_18tasks'],
    suiteLabels: { rlbench_18tasks: '18-Task Multi-Task' },
  },
  maniskill: {
    key: 'maniskill_avg',
    label: 'ManiSkill',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['maniskill_pick_and_place', 'maniskill_assembly'],
    suiteLabels: { maniskill_pick_and_place: 'Pick & Place', maniskill_assembly: 'Assembly' },
  },
  vlabench: {
    key: 'vlabench_avg',
    label: 'VLABench',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['vlabench_primitive', 'vlabench_composite'],
    suiteLabels: { vlabench_primitive: 'Primitive', vlabench_composite: 'Composite' },
  },
  robocasa: {
    key: 'robocasa_avg',
    label: 'RoboCasa',
    max: 100,
    metric: 'Success Rate (%)',
    suites: ['robocasa_pick_and_place', 'robocasa_open_close'],
    suiteLabels: { robocasa_pick_and_place: 'Pick & Place', robocasa_open_close: 'Open/Close' },
  },
}

/**
 * Flat array of { key, label, max } for heatmaps, radar charts, and coverage tables.
 * Derived from BENCHMARKS so it stays in sync automatically.
 */
export const BENCHMARK_LIST = Object.values(BENCHMARKS).map(({ key, label, max }) => ({ key, label, max }))

/**
 * Shorthand label mapping for compact displays (e.g. heatmap column headers).
 */
export const BENCHMARK_SHORT_LABELS = {
  robotwin_v1_avg: 'RTwin v1',
  robotwin_v2_avg: 'RTwin v2',
}

/**
 * Get a short label for a benchmark key, falling back to the full label.
 */
export function getBenchmarkShortLabel(key) {
  if (BENCHMARK_SHORT_LABELS[key]) return BENCHMARK_SHORT_LABELS[key]
  const bench = BENCHMARK_LIST.find(b => b.key === key)
  return bench?.label ?? key
}

/**
 * Shared color palettes used across multiple components.
 */
export const COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#3498DB', '#F39C12', '#E74C3C', '#2ECC71']

export const ACTION_HEAD_COLORS = {
  'flow matching': '#7F77DD',
  'autoregressive': '#1D9E75',
  'diffusion': '#D85A30',
  'diffusion transformer': '#D4537E',
  'chain-of-thought': '#F39C12',
  'parallel decoding': '#3498DB',
  'hybrid': '#E67E22',
  'fast tokenizer': '#2ECC71',
  'unknown': '#71717a',
}

export const VENUE_COLORS = {
  'ICML': '#E74C3C',
  'NeurIPS': '#3498DB',
  'ICLR': '#2ECC71',
  'CoRL': '#9B59B6',
  'RSS': '#F39C12',
  'ICRA': '#1ABC9C',
  'CVPR': '#E67E22',
  'RA-L': '#7F8C8D',
}

export const EVAL_COLORS = {
  'fine-tuned': { bg: 'bg-blue-500/10', text: 'text-blue-400', label: 'FT' },
  'zero-shot': { bg: 'bg-amber-500/10', text: 'text-amber-400', label: 'ZS' },
}

/**
 * Shared utility: heat color for benchmark scores.
 */
export function heatColor(value, max) {
  if (value == null) return 'transparent'
  const pct = Math.min(value / max, 1)
  if (pct >= 0.9) return 'rgba(16, 185, 129, 0.7)'
  if (pct >= 0.75) return 'rgba(16, 185, 129, 0.5)'
  if (pct >= 0.6) return 'rgba(250, 204, 21, 0.5)'
  if (pct >= 0.4) return 'rgba(249, 115, 22, 0.5)'
  if (pct >= 0.2) return 'rgba(239, 68, 68, 0.5)'
  return 'rgba(239, 68, 68, 0.3)'
}
