import { useState, useMemo } from 'react'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'

const COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#3498DB', '#F39C12', '#E74C3C', '#2ECC71']

// All benchmark suites for radar normalization (0-100 scale)
const RADAR_AXES = [
  { key: 'libero_avg', label: 'LIBERO', max: 100 },
  { key: 'calvin_avg', label: 'CALVIN', max: 5 },
  { key: 'simpler_avg', label: 'SimplerEnv', max: 100 },
  { key: 'robotwin_v1_avg', label: 'RoboTwin v1', max: 100 },
  { key: 'robotwin_v2_avg', label: 'RoboTwin v2', max: 100 },
  { key: 'rlbench_avg', label: 'RLBench', max: 100 },
  { key: 'maniskill_avg', label: 'ManiSkill', max: 100 },
  { key: 'vlabench_avg', label: 'VLABench', max: 100 },
  { key: 'robocasa_avg', label: 'RoboCasa', max: 100 },
]

const DETAIL_SUITES = {
  libero: {
    label: 'LIBERO',
    avgKey: 'libero_avg',
    suites: [
      { key: 'libero_spatial', label: 'Spatial' },
      { key: 'libero_object', label: 'Object' },
      { key: 'libero_goal', label: 'Goal' },
      { key: 'libero_long', label: 'Long' },
    ],
  },
  calvin: {
    label: 'CALVIN',
    avgKey: 'calvin_avg',
    suites: [{ key: 'calvin_abc_d_avg_len', label: 'ABC→D Avg Len' }],
  },
  robotwin_v1: {
    label: 'RoboTwin v1',
    avgKey: 'robotwin_v1_avg',
    suites: [
      { key: 'robotwin_easy', label: 'Easy' },
      { key: 'robotwin_hard', label: 'Hard' },
    ],
  },
  robotwin_v2: {
    label: 'RoboTwin v2',
    avgKey: 'robotwin_v2_avg',
    suites: [
      { key: 'short_horizon', label: 'Short' },
      { key: 'medium_horizon', label: 'Medium' },
      { key: 'long_horizon', label: 'Long' },
    ],
  },
  vlabench: {
    label: 'VLABench',
    avgKey: 'vlabench_avg',
    suites: [
      { key: 'vlabench_primitive', label: 'Primitive' },
      { key: 'vlabench_composite', label: 'Composite' },
    ],
  },
}

function normalize(value, max) {
  if (value == null) return 0
  return Math.round((value / max) * 100)
}

export default function ModelCompare({ models }) {
  const [selected, setSelected] = useState([])
  const [searchQuery, setSearchQuery] = useState('')

  // Models sorted by total benchmark coverage then by libero_avg
  const sortedModels = useMemo(() => {
    return [...models].sort((a, b) => {
      const coverA = RADAR_AXES.filter(ax => a[ax.key] != null).length
      const coverB = RADAR_AXES.filter(ax => b[ax.key] != null).length
      if (coverB !== coverA) return coverB - coverA
      return (b.libero_avg ?? -1) - (a.libero_avg ?? -1)
    })
  }, [models])

  const filteredModels = useMemo(() => {
    if (!searchQuery) return sortedModels.slice(0, 20)
    const q = searchQuery.toLowerCase()
    return sortedModels.filter(m =>
      m.name.toLowerCase().includes(q) || m.organization?.toLowerCase().includes(q)
    ).slice(0, 20)
  }, [sortedModels, searchQuery])

  const selectedModels = useMemo(() => {
    return selected.map(name => models.find(m => m.name === name)).filter(Boolean)
  }, [selected, models])

  const toggleModel = (name) => {
    setSelected(prev =>
      prev.includes(name)
        ? prev.filter(n => n !== name)
        : prev.length < 8 ? [...prev, name] : prev
    )
  }

  // Quick presets
  const presets = useMemo(() => {
    const top3Libero = [...models]
      .filter(m => m.libero_avg != null)
      .sort((a, b) => b.libero_avg - a.libero_avg)
      .slice(0, 3).map(m => m.name)
    const top3Calvin = [...models]
      .filter(m => m.calvin_avg != null)
      .sort((a, b) => b.calvin_avg - a.calvin_avg)
      .slice(0, 3).map(m => m.name)
    const ossModels = [...models]
      .filter(m => m.open_source && m.libero_avg != null)
      .sort((a, b) => b.libero_avg - a.libero_avg)
      .slice(0, 3).map(m => m.name)
    return [
      { label: 'Top 3 LIBERO', models: top3Libero },
      { label: 'Top 3 CALVIN', models: top3Calvin },
      { label: 'Top 3 Open-Source', models: ossModels },
    ]
  }, [models])

  // Radar data
  const radarData = useMemo(() => {
    return RADAR_AXES.map(ax => {
      const point = { axis: ax.label }
      selectedModels.forEach(m => {
        point[m.name] = normalize(m[ax.key], ax.max)
      })
      return point
    })
  }, [selectedModels])

  // Head-to-head detail bar data per benchmark
  const detailData = useMemo(() => {
    const results = {}
    for (const [benchKey, cfg] of Object.entries(DETAIL_SUITES)) {
      const hasData = selectedModels.some(m => m.benchmarks?.[benchKey])
      if (!hasData) continue
      results[benchKey] = cfg.suites.map(s => {
        const point = { suite: s.label }
        selectedModels.forEach(m => {
          point[m.name] = m.benchmarks?.[benchKey]?.[s.key] ?? null
        })
        return point
      })
    }
    return results
  }, [selectedModels])

  // Stats comparison table
  const statsRows = useMemo(() => {
    if (selectedModels.length === 0) return []
    return [
      { label: 'Organization', get: m => m.organization || '—' },
      { label: 'Parameters', get: m => m.architecture?.parameters || '—' },
      { label: 'Action Head', get: m => m.architecture?.action_head || '—' },
      { label: 'Inference Hz', get: m => m.inference_hz ? `${m.inference_hz} Hz` : '—' },
      { label: 'Open Source', get: m => m.open_source ? '✓' : '✗' },
      { label: 'Venue', get: m => m.venue || 'preprint' },
      { label: 'Date', get: m => m.date || '—' },
    ]
  }, [selectedModels])

  return (
    <div>
      {/* Model selector */}
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-white mb-2">Select Models to Compare (max 8)</h3>

        {/* Presets */}
        <div className="flex gap-1.5 mb-2 flex-wrap">
          {presets.map(p => (
            <button
              key={p.label}
              onClick={() => setSelected(p.models)}
              className="px-2.5 py-1 text-[11px] rounded-md border border-zinc-700/50 text-zinc-400 hover:border-purple-500/50 hover:text-purple-300 transition-all"
            >
              {p.label}
            </button>
          ))}
          {selected.length > 0 && (
            <button
              onClick={() => setSelected([])}
              className="px-2.5 py-1 text-[11px] rounded-md border border-red-500/30 text-red-400 hover:bg-red-500/10 transition-all"
            >
              Clear
            </button>
          )}
        </div>

        {/* Search */}
        <input
          type="text"
          placeholder="Search models..."
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="w-full px-3 py-1.5 text-xs rounded-md border border-zinc-700 bg-zinc-900 text-white placeholder-zinc-500 focus:outline-none focus:border-purple-500/50 mb-2"
        />

        {/* Model chips */}
        <div className="flex gap-1.5 flex-wrap max-h-[120px] overflow-y-auto">
          {filteredModels.map(m => {
            const isSelected = selected.includes(m.name)
            const colorIdx = selected.indexOf(m.name)
            return (
              <button
                key={m.name}
                onClick={() => toggleModel(m.name)}
                className={`px-2 py-1 text-[11px] rounded-md border transition-all flex items-center gap-1 ${
                  isSelected
                    ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                    : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-500'
                }`}
              >
                {isSelected && (
                  <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: COLORS[colorIdx % COLORS.length] }} />
                )}
                {m.name}
              </button>
            )
          })}
        </div>
      </div>

      {selectedModels.length === 0 ? (
        <div className="text-center py-16 text-zinc-500 text-sm">
          Select models above to compare
        </div>
      ) : (
        <div className="space-y-6">
          {/* Radar Chart */}
          <div>
            <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
              Overall Benchmark Profile
            </h4>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="#27272a" />
                <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11, fill: '#a1a1aa' }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#52525b' }} />
                <Tooltip
                  contentStyle={{
                    fontSize: 11,
                    borderRadius: 8,
                    border: '1px solid #3f3f46',
                    backgroundColor: '#18181b',
                    color: '#e5e5e5',
                  }}
                />
                {selectedModels.map((m, i) => (
                  <Radar
                    key={m.name}
                    name={m.name}
                    dataKey={m.name}
                    stroke={COLORS[i % COLORS.length]}
                    fill={COLORS[i % COLORS.length]}
                    fillOpacity={0.1}
                    strokeWidth={2}
                  />
                ))}
              </RadarChart>
            </ResponsiveContainer>
            <div className="flex gap-3 justify-center flex-wrap">
              {selectedModels.map((m, i) => (
                <span key={m.name} className="flex items-center gap-1.5 text-[11px] text-zinc-400">
                  <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                  {m.name}
                </span>
              ))}
            </div>
            <p className="text-[10px] text-zinc-600 text-center mt-1">Scores normalized to 0-100 scale. CALVIN (max 5) scaled proportionally.</p>
          </div>

          {/* Spec Comparison Table */}
          <div>
            <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
              Specifications
            </h4>
            <div className="border border-zinc-800 rounded-xl overflow-hidden overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-zinc-900/50">
                    <th className="px-3 py-2 text-left text-zinc-500 font-medium w-28"></th>
                    {selectedModels.map((m, i) => (
                      <th key={m.name} className="px-3 py-2 text-center font-semibold" style={{ color: COLORS[i % COLORS.length] }}>
                        {m.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {statsRows.map(row => (
                    <tr key={row.label} className="border-t border-zinc-800/50">
                      <td className="px-3 py-2 text-zinc-500 font-medium">{row.label}</td>
                      {selectedModels.map(m => (
                        <td key={m.name} className="px-3 py-2 text-center text-zinc-300">
                          {row.get(m)}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {/* Benchmark scores */}
                  {RADAR_AXES.map(ax => (
                    <tr key={ax.key} className="border-t border-zinc-800/50">
                      <td className="px-3 py-2 text-zinc-500 font-medium">{ax.label}</td>
                      {selectedModels.map(m => {
                        const val = m[ax.key]
                        const allVals = selectedModels.map(mm => mm[ax.key]).filter(v => v != null)
                        const isMax = val != null && allVals.length > 1 && val >= Math.max(...allVals)
                        return (
                          <td key={m.name} className={`px-3 py-2 text-center tabular-nums ${isMax ? 'text-emerald-400 font-bold' : val != null ? 'text-zinc-300' : 'text-zinc-600'}`}>
                            {val != null ? (ax.max === 5 ? val.toFixed(1) : val.toFixed(1)) : '—'}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Detail bar charts per benchmark */}
          {Object.entries(detailData).map(([benchKey, data]) => (
            <div key={benchKey}>
              <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                {DETAIL_SUITES[benchKey].label} — Suite Breakdown
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="suite" tick={{ fontSize: 11, fill: '#a1a1aa' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#71717a' }} />
                  <Tooltip
                    contentStyle={{
                      fontSize: 11,
                      borderRadius: 8,
                      border: '1px solid #3f3f46',
                      backgroundColor: '#18181b',
                      color: '#e5e5e5',
                    }}
                  />
                  {selectedModels.map((m, i) => (
                    <Bar key={m.name} dataKey={m.name} fill={COLORS[i % COLORS.length]} radius={[3, 3, 0, 0]} barSize={20} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
