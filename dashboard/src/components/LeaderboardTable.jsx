import { useState, useMemo } from 'react'
import { BENCHMARKS, ACTION_HEAD_COLORS, VENUE_COLORS, EVAL_COLORS, classifyEvalCondition } from '../constants/benchmarks'

function getVenueColor(venue) {
  if (!venue) return null
  for (const [key, color] of Object.entries(VENUE_COLORS)) {
    if (venue.includes(key)) return color
  }
  return '#7F8C8D'
}

function getActionColor(actionHead) {
  const lower = (actionHead || '').toLowerCase()
  for (const [key, color] of Object.entries(ACTION_HEAD_COLORS)) {
    if (lower.includes(key)) return color
  }
  return '#888'
}

function formatDate(dateStr) {
  if (!dateStr) return '—'
  const d = new Date(dateStr + 'T00:00:00Z')
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
  return `${months[d.getUTCMonth()]} ${d.getUTCFullYear()}`
}

function getEvalStyle(model, benchKey) {
  const cond = model.eval_conditions?.[benchKey]
  if (!cond) return null
  return classifyEvalCondition(cond)
}

const MODEL_TYPE_FILTERS = {
  all: { label: 'All' },
  video: { label: 'Video Models' },
}

export default function LeaderboardTable({ models, onModelClick }) {
  const [activeBench, setActiveBench] = useState('libero')
  const [sortBy, setSortBy] = useState('avg')
  const [typeFilter, setTypeFilter] = useState('all')

  const bench = BENCHMARKS[activeBench]

  const sorted = useMemo(() => {
    return [...models]
      .filter(m => m.benchmarks?.[activeBench])
      .filter(m => typeFilter === 'all' || m.model_type === typeFilter)
      .map(m => {
        const scores = m.benchmarks?.[activeBench] || {}
        const avg = m[bench.key]
        return { ...m, _scores: scores, _avg: avg }
      })
      .sort((a, b) => {
        if (sortBy === 'avg') return (b._avg ?? -1) - (a._avg ?? -1)
        return (b._scores?.[sortBy] ?? -1) - (a._scores?.[sortBy] ?? -1)
      })
  }, [models, activeBench, sortBy, bench.key, typeFilter])

  const medals = ['🥇', '🥈', '🥉']

  return (
    <div>
      {/* Benchmark selector */}
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(BENCHMARKS).map(([key, b]) => {
          const count = models.filter(m => m.benchmarks?.[key]).length
          return (
            <button
              key={key}
              onClick={() => { setActiveBench(key); setSortBy('avg') }}
              className={`px-3 py-1.5 text-xs rounded-md border transition-all ${
                activeBench === key
                  ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                  : 'border-zinc-700 text-zinc-400 hover:border-zinc-500'
              }`}
            >
              {b.label}
              <span className="ml-1.5 text-[10px] opacity-60">({count})</span>
            </button>
          )
        })}
      </div>

      {/* Model type filter */}
      <div className="flex gap-1.5 mb-3 flex-wrap">
        {Object.entries(MODEL_TYPE_FILTERS).map(([key, cfg]) => (
          <button
            key={key}
            onClick={() => setTypeFilter(key)}
            className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
              typeFilter === key
                ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-300 font-semibold'
                : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
            }`}
          >
            {cfg.label}
          </button>
        ))}
      </div>

      {/* Suite sort buttons */}
      <div className="flex gap-1.5 mb-4 flex-wrap">
        <button
          onClick={() => setSortBy('avg')}
          className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
            sortBy === 'avg'
              ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
              : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
          }`}
        >
          Average
        </button>
        {bench.suites.map(key => (
          <button
            key={key}
            onClick={() => setSortBy(key)}
            className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
              sortBy === key
                ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
                : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
            }`}
          >
            {bench.suiteLabels[key]}
          </button>
        ))}
      </div>

      {/* Table */}
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-zinc-900/50">
              <th className="px-3 py-3 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider w-8">#</th>
              <th className="px-3 py-3 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Model</th>
              <th className="px-3 py-3 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider hidden md:table-cell">Date</th>
              <th className="px-3 py-3 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider hidden sm:table-cell">Action Head</th>
              <th className="px-3 py-3 text-left text-[10px] font-semibold text-zinc-500 uppercase tracking-wider hidden lg:table-cell">Venue</th>
              <th className="px-3 py-3 text-center text-[10px] font-semibold text-zinc-500 uppercase tracking-wider w-10 hidden sm:table-cell" title="Evaluation Condition">Eval</th>
              <th className="px-3 py-3 text-right text-[10px] font-semibold text-zinc-500 uppercase tracking-wider hidden lg:table-cell" title="Inference Speed (Hz)">Hz</th>
              <th className="px-3 py-3 text-right text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">
                {sortBy === 'avg' ? 'Avg' : bench.suiteLabels[sortBy]}
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => {
              const score = sortBy === 'avg'
                ? m._avg?.toFixed(1)
                : m._scores?.[sortBy]?.toFixed(1)

              const evalStyle = getEvalStyle(m, activeBench)

              return (
                <tr key={m.name} className="border-t border-zinc-800/50 hover:bg-zinc-900/30 transition-colors">
                  <td className="px-3 py-3 font-medium">
                    {i < 3 ? medals[i] : <span className="text-zinc-500">{i + 1}</span>}
                  </td>
                  <td className="px-3 py-3">
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => onModelClick?.(m)}
                        className="font-semibold text-white hover:text-blue-400 transition-colors text-left cursor-pointer"
                      >
                        {m.name}
                      </button>
                      {m.open_source && (
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-medium">OSS</span>
                      )}
                      {m.code_url && (
                        <a href={m.code_url} target="_blank" rel="noopener noreferrer" className="text-[9px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-medium hover:bg-blue-500/20 transition-colors">Code</a>
                      )}
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] text-zinc-500">{m.organization}</span>
                      {(() => {
                        const count = Object.keys(m.benchmarks || {}).length
                        return (
                          <span className={`text-[9px] tabular-nums ${count >= 3 ? 'text-emerald-600' : count >= 1 ? 'text-zinc-600' : 'text-zinc-700'}`}>
                            {count}/{Object.keys(BENCHMARKS).length}
                          </span>
                        )
                      })()}
                    </div>
                  </td>
                  <td className="px-3 py-3 text-xs text-zinc-400 hidden md:table-cell">
                    {formatDate(m.date)}
                  </td>
                  <td className="px-3 py-3 hidden sm:table-cell">
                    <span
                      className="inline-block px-2 py-0.5 text-[10px] font-medium rounded-md"
                      style={{
                        backgroundColor: getActionColor(m.architecture?.action_head) + '18',
                        color: getActionColor(m.architecture?.action_head),
                      }}
                    >
                      {m.architecture?.action_head || 'unknown'}
                    </span>
                  </td>
                  <td className="px-3 py-3 hidden lg:table-cell">
                    {m.venue ? (
                      <span
                        className="inline-block px-2 py-0.5 text-[10px] font-medium rounded-md"
                        style={{
                          backgroundColor: getVenueColor(m.venue) + '18',
                          color: getVenueColor(m.venue),
                        }}
                      >
                        {m.venue}
                      </span>
                    ) : (
                      <span className="text-zinc-600 text-[10px]">preprint</span>
                    )}
                  </td>
                  <td className="px-3 py-3 text-center hidden sm:table-cell">
                    {evalStyle ? (
                      <span className={`inline-block px-1.5 py-0.5 text-[9px] font-semibold rounded ${evalStyle.bg} ${evalStyle.text}`}
                        title={m.eval_conditions?.[activeBench]}
                      >
                        {evalStyle.label}
                      </span>
                    ) : (
                      <span className="text-zinc-600 text-[10px]">—</span>
                    )}
                  </td>
                  <td className="px-3 py-3 text-right text-xs tabular-nums hidden lg:table-cell">
                    {m.inference_hz ? (
                      <span
                        className={`font-medium ${m.inference_hz >= 30 ? 'text-emerald-400' : m.inference_hz >= 10 ? 'text-amber-400' : 'text-red-400'}`}
                        title={m.inference_hz >= 30 ? 'Fast (≥30Hz)' : m.inference_hz >= 10 ? 'Medium (≥10Hz)' : 'Slow (<10Hz)'}
                      >
                        {m.inference_hz}{m.inference_hz >= 30 ? '↑' : m.inference_hz < 10 ? '↓' : ''}
                      </span>
                    ) : (
                      <span className="text-zinc-600">—</span>
                    )}
                  </td>
                  <td className={`px-3 py-3 text-right font-bold text-base tabular-nums ${
                    i === 0 ? 'text-emerald-400' : 'text-white'
                  }`}>
                    {score ?? 'N/A'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Footer info */}
      <div className="flex items-center justify-between mt-2 text-[11px] text-zinc-600">
        <span>
          Benchmark: {bench.label} · {bench.metric} · Higher is better
        </span>
        <span className="flex items-center gap-2">
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-400" />FT = Fine-tuned
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-rose-400" />RL = RL-trained
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-400" />ZS = Zero-shot
          </span>
        </span>
      </div>

      {/* Fairness note */}
      <div className="mt-3 px-3 py-2 bg-zinc-900/50 border border-zinc-800/50 rounded-lg text-[11px] text-zinc-500 leading-relaxed">
        <strong className="text-zinc-400">Note on fairness:</strong> Models are evaluated under different conditions
        (fine-tuned vs zero-shot, different training data, compute budgets). The "Eval" column shows the evaluation
        condition. Direct ranking comparisons should be interpreted with caution.
      </div>
    </div>
  )
}
