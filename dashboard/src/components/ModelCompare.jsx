import { useState, useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const BENCH_CONFIGS = {
  libero: {
    label: 'LIBERO',
    suites: { libero_spatial: 'Spatial', libero_object: 'Object', libero_goal: 'Goal', libero_long: 'Long' },
    avgKey: 'libero_avg',
    domain: [40, 100],
  },
  calvin: {
    label: 'CALVIN',
    suites: { calvin_abc_d_avg_len: 'ABC→D' },
    avgKey: 'calvin_avg',
    domain: [0, 5],
  },
  simpler_env: {
    label: 'SimplerEnv',
    suites: { google_robot_pick_coke_can: 'Pick Can', google_robot_move_near: 'Move Near' },
    avgKey: 'simpler_avg',
    domain: [0, 100],
  },
  robotwin_v1: {
    label: 'RoboTwin v1',
    suites: { robotwin_easy: 'Easy', robotwin_hard: 'Hard' },
    avgKey: 'robotwin_v1_avg',
    domain: [0, 100],
  },
  robotwin_v2: {
    label: 'RoboTwin v2',
    suites: { short_horizon: 'Short', medium_horizon: 'Medium', long_horizon: 'Long' },
    avgKey: 'robotwin_v2_avg',
    domain: [0, 100],
  },
}

const TOP_COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#3498DB']

export default function ModelCompare({ models }) {
  const [benchKey, setBenchKey] = useState('libero')
  const config = BENCH_CONFIGS[benchKey]

  const topModels = useMemo(() => {
    return [...models]
      .filter(m => m.benchmarks?.[benchKey] && m[config.avgKey] != null)
      .sort((a, b) => b[config.avgKey] - a[config.avgKey])
      .slice(0, 5)
  }, [models, benchKey, config.avgKey])

  const barData = useMemo(() => {
    return Object.entries(config.suites).map(([key, label]) => {
      const point = { suite: label }
      topModels.forEach(m => {
        point[m.name] = m.benchmarks[benchKey]?.[key]
      })
      return point
    })
  }, [topModels, benchKey, config.suites])

  // Compute per-model insights
  const insights = useMemo(() => {
    if (benchKey !== 'libero' || topModels.length < 2) return null
    const top = topModels[0]
    const spatial = top.benchmarks.libero?.libero_spatial
    const long = top.benchmarks.libero?.libero_long
    if (!spatial || !long) return null
    const gap = (spatial - long).toFixed(1)

    // Find model with smallest Spatial-Long gap
    let smallestGapModel = null
    let smallestGap = Infinity
    for (const m of topModels) {
      const s = m.benchmarks.libero?.libero_spatial
      const l = m.benchmarks.libero?.libero_long
      if (s && l) {
        const g = s - l
        if (g < smallestGap) {
          smallestGap = g
          smallestGapModel = m.name
        }
      }
    }

    return {
      topName: top.name,
      spatial,
      long,
      gap,
      smallestGapModel,
      smallestGap: smallestGap.toFixed(1),
    }
  }, [topModels, benchKey])

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white">Cross-Suite Comparison</h3>
        <div className="flex gap-1.5">
          {Object.entries(BENCH_CONFIGS).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setBenchKey(key)}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                benchKey === key
                  ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              {cfg.label}
            </button>
          ))}
        </div>
      </div>

      {topModels.length === 0 ? (
        <div className="text-center py-12 text-zinc-500 text-sm">
          No models with {config.label} data
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="suite" tick={{ fontSize: 12, fill: '#a1a1aa' }} />
              <YAxis domain={config.domain} tick={{ fontSize: 11, fill: '#71717a' }} />
              <Tooltip
                contentStyle={{
                  fontSize: 12,
                  borderRadius: 8,
                  border: '1px solid #3f3f46',
                  backgroundColor: '#18181b',
                  color: '#e5e5e5',
                }}
              />
              {topModels.map((m, i) => (
                <Bar
                  key={m.name}
                  dataKey={m.name}
                  fill={TOP_COLORS[i % TOP_COLORS.length]}
                  radius={[4, 4, 0, 0]}
                  barSize={16}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>

          <div className="flex gap-3 justify-center mt-2 flex-wrap">
            {topModels.map((m, i) => (
              <span key={m.name} className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ backgroundColor: TOP_COLORS[i % TOP_COLORS.length] }} />
                {m.name}
                <span className="text-zinc-600 text-[10px]">({m.date?.slice(0, 4) || '?'})</span>
              </span>
            ))}
          </div>

          {insights && (
            <div className="mt-4 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg text-xs text-amber-200/80">
              <strong>AI Insight:</strong> LIBERO-Long remains the most challenging suite.
              {insights.topName} scores {insights.long}% on Long vs {insights.spatial}% on Spatial ({insights.gap}%p gap).
              {insights.smallestGapModel !== insights.topName && (
                <> {insights.smallestGapModel} shows the smallest gap at {insights.smallestGap}%p, suggesting its approach may be better suited for long-horizon sequential tasks.</>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
