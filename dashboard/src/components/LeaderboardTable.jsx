import { useState, useMemo } from 'react'

const LIBERO_SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long']
const SUITE_LABELS = {
  avg: 'Average',
  libero_spatial: 'Spatial',
  libero_object: 'Object',
  libero_goal: 'Goal',
  libero_long: 'Long',
}

const ACTION_HEAD_COLORS = {
  'flow matching': '#7F77DD',
  'autoregressive': '#1D9E75',
  'diffusion': '#D85A30',
  'chain-of-thought': '#D4537E',
  'parallel decoding': '#3498DB',
  'hybrid': '#E67E22',
  'FAST tokenizer': '#2ECC71',
}

function getActionColor(actionHead) {
  const lower = (actionHead || '').toLowerCase()
  for (const [key, color] of Object.entries(ACTION_HEAD_COLORS)) {
    if (lower.includes(key)) return color
  }
  return '#888'
}

export default function LeaderboardTable({ models }) {
  const [sortBy, setSortBy] = useState('avg')

  const sorted = useMemo(() => {
    return [...models]
      .filter(m => m.benchmarks?.libero)
      .map(m => {
        const lib = m.benchmarks.libero
        const scores = LIBERO_SUITES.map(s => lib[s]).filter(v => v != null)
        const avg = scores.length === 4 ? scores.reduce((a, b) => a + b, 0) / 4 : null
        return { ...m, libero_scores: lib, libero_avg: avg }
      })
      .sort((a, b) => {
        if (sortBy === 'avg') return (b.libero_avg ?? -1) - (a.libero_avg ?? -1)
        return (b.libero_scores?.[sortBy] ?? -1) - (a.libero_scores?.[sortBy] ?? -1)
      })
  }, [models, sortBy])

  const medals = ['🥇', '🥈', '🥉']

  return (
    <div>
      <div className="flex gap-2 mb-4 flex-wrap">
        {Object.entries(SUITE_LABELS).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setSortBy(key)}
            className={`px-3 py-1.5 text-xs rounded-md border transition-all ${
              sortBy === key
                ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
                : 'border-zinc-700 text-zinc-400 hover:border-zinc-500'
            }`}
          >
            {key === 'avg' ? label : `LIBERO-${label}`}
          </button>
        ))}
      </div>

      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-900/50">
              <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider w-10">#</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider">Model</th>
              <th className="px-4 py-3 text-left text-xs font-semibold text-zinc-500 uppercase tracking-wider hidden sm:table-cell">Action Head</th>
              <th className="px-4 py-3 text-right text-xs font-semibold text-zinc-500 uppercase tracking-wider">
                {sortBy === 'avg' ? 'Avg' : SUITE_LABELS[sortBy]}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => {
              const score = sortBy === 'avg'
                ? m.libero_avg?.toFixed(1)
                : m.libero_scores?.[sortBy]?.toFixed(1)

              return (
                <tr key={m.name} className="border-t border-zinc-800/50 hover:bg-zinc-900/30 transition-colors">
                  <td className="px-4 py-3 font-medium">
                    {i < 3 ? medals[i] : <span className="text-zinc-500">{i + 1}</span>}
                  </td>
                  <td className="px-4 py-3">
                    <div className="font-semibold text-white">{m.name}</div>
                    <div className="text-xs text-zinc-500">{m.organization}</div>
                  </td>
                  <td className="px-4 py-3 hidden sm:table-cell">
                    <span
                      className="inline-block px-2 py-0.5 text-[11px] font-medium rounded-md"
                      style={{
                        backgroundColor: getActionColor(m.architecture?.action_head) + '18',
                        color: getActionColor(m.architecture?.action_head),
                      }}
                    >
                      {m.architecture?.action_head || 'unknown'}
                    </span>
                  </td>
                  <td className={`px-4 py-3 text-right font-bold text-base tabular-nums ${
                    i === 0 ? 'text-emerald-400' : 'text-white'
                  }`}>
                    {score ? `${score}%` : 'N/A'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <p className="text-[11px] text-zinc-600 mt-2">
        Benchmark: LIBERO · Metric: Success Rate (%) · Higher is better · Last updated: 2026-03-15
      </p>
    </div>
  )
}
