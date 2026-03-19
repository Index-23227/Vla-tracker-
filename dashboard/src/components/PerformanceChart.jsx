import { useState, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const BENCH_OPTIONS = {
  libero: { label: 'LIBERO', avgKey: 'libero_avg', domain: [60, 100] },
  calvin: { label: 'CALVIN', avgKey: 'calvin_avg', domain: [0, 5] },
  robotwin_v1: { label: 'RoboTwin v1', avgKey: 'robotwin_v1_avg', domain: [0, 100] },
  robotwin_v2: { label: 'RoboTwin v2', avgKey: 'robotwin_v2_avg', domain: [0, 100] },
  maniskill: { label: 'ManiSkill', avgKey: 'maniskill_avg', domain: [0, 100] },
  vlabench: { label: 'VLABench', avgKey: 'vlabench_avg', domain: [0, 100] },
  robocasa: { label: 'RoboCasa', avgKey: 'robocasa_avg', domain: [0, 100] },
}

export default function PerformanceChart({ modelHistory, models }) {
  const [benchKey, setBenchKey] = useState('libero')
  const benchCfg = BENCH_OPTIONS[benchKey]

  // For non-LIBERO benchmarks, generate history from current snapshot
  const effectiveHistory = useMemo(() => {
    if (benchKey === 'libero') return modelHistory

    // Build a simple "history" from current model data for other benchmarks
    if (!models) return []
    const relevantModels = models
      .filter(m => m[benchCfg.avgKey] != null)
      .sort((a, b) => b[benchCfg.avgKey] - a[benchCfg.avgKey])
      .slice(0, 10)

    const COLORS = ['#E74C3C', '#7F77DD', '#2ECC71', '#F39C12', '#1D9E75', '#D85A30', '#9B59B6', '#8E44AD', '#888780', '#639922']
    return relevantModels.map((m, i) => ({
      name: m.name,
      color: COLORS[i % COLORS.length],
      history: [{ month: 'Current', avg: m[benchCfg.avgKey] }],
    }))
  }, [benchKey, modelHistory, models, benchCfg.avgKey])

  const allModels = effectiveHistory.map(m => m.name)
  const [selected, setSelected] = useState(null)

  // Reset selection when benchmark changes
  const activeSelected = useMemo(() => {
    if (selected !== null && selected.every(s => allModels.includes(s))) return selected
    return allModels.slice(0, 4)
  }, [selected, allModels])

  const chartData = useMemo(() => {
    if (benchKey === 'libero') {
      const months = ['Oct 24', 'Jan 25', 'Apr 25', 'Jul 25', 'Oct 25', 'Jan 26']
      return months.map(month => {
        const point = { month }
        effectiveHistory
          .filter(m => activeSelected.includes(m.name))
          .forEach(m => {
            const h = m.history.find(h => h.month === month)
            if (h && h.avg != null) point[m.name] = h.avg
          })
        return point
      })
    }

    // For other benchmarks, show bar-like single-point data
    const point = { month: 'Current' }
    effectiveHistory
      .filter(m => activeSelected.includes(m.name))
      .forEach(m => {
        const h = m.history[0]
        if (h && h.avg != null) point[m.name] = h.avg
      })
    return [point]
  }, [activeSelected, effectiveHistory, benchKey])

  const toggle = (name) => {
    setSelected(prev => {
      const cur = prev !== null ? prev : activeSelected
      return cur.includes(name) ? cur.filter(n => n !== name) : [...cur, name]
    })
  }

  return (
    <div>
      {/* Benchmark selector */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-white">
          {benchCfg.label} Score {benchKey === 'libero' ? 'Over Time' : 'Comparison'}
        </h3>
        <div className="flex gap-1.5">
          {Object.entries(BENCH_OPTIONS).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => { setBenchKey(key); setSelected(null) }}
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

      <div className="flex gap-3 mb-4 flex-wrap">
        {effectiveHistory.map(m => (
          <label key={m.name} className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={activeSelected.includes(m.name)}
              onChange={() => toggle(m.name)}
              className="accent-purple-500"
            />
            <span
              className="w-2.5 h-2.5 rounded-sm inline-block"
              style={{ backgroundColor: m.color }}
            />
            {m.name}
          </label>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="month" tick={{ fontSize: 11, fill: '#71717a' }} />
          <YAxis domain={benchCfg.domain} tick={{ fontSize: 11, fill: '#71717a' }} />
          <Tooltip
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: '1px solid #3f3f46',
              backgroundColor: '#18181b',
              color: '#e5e5e5',
            }}
          />
          {effectiveHistory
            .filter(m => activeSelected.includes(m.name))
            .map(m => (
              <Line
                key={m.name}
                type="monotone"
                dataKey={m.name}
                stroke={m.color}
                strokeWidth={2}
                dot={{ r: 3 }}
                connectNulls={false}
              />
            ))}
        </LineChart>
      </ResponsiveContainer>

      <p className="text-[11px] text-zinc-600 mt-1">
        {benchKey === 'libero'
          ? 'Models appear when first published. Gap = not yet released.'
          : 'Showing current scores for models with available data.'}
      </p>
    </div>
  )
}
