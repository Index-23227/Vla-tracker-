import { useState, useMemo } from 'react'

const ALL_BENCHMARKS = [
  { key: 'libero_avg', label: 'LIBERO', max: 100 },
  { key: 'calvin_avg', label: 'CALVIN', max: 5 },
  { key: 'simpler_avg', label: 'SimplerEnv', max: 100 },
  { key: 'robotwin_v1_avg', label: 'RTwin v1', max: 100 },
  { key: 'robotwin_v2_avg', label: 'RTwin v2', max: 100 },
  { key: 'rlbench_avg', label: 'RLBench', max: 100 },
  { key: 'maniskill_avg', label: 'ManiSkill', max: 100 },
  { key: 'vlabench_avg', label: 'VLABench', max: 100 },
  { key: 'robocasa_avg', label: 'RoboCasa', max: 100 },
]

function heatColor(value, max) {
  if (value == null) return 'transparent'
  const pct = Math.min(value / max, 1)
  if (pct >= 0.9) return 'rgba(16, 185, 129, 0.8)'
  if (pct >= 0.75) return 'rgba(16, 185, 129, 0.5)'
  if (pct >= 0.6) return 'rgba(250, 204, 21, 0.5)'
  if (pct >= 0.4) return 'rgba(249, 115, 22, 0.5)'
  if (pct >= 0.2) return 'rgba(239, 68, 68, 0.5)'
  return 'rgba(239, 68, 68, 0.3)'
}

const SORT_OPTIONS = [
  { key: 'coverage', label: 'Coverage' },
  { key: 'libero_avg', label: 'LIBERO' },
  { key: 'name', label: 'Name' },
  { key: 'date', label: 'Date' },
]

export default function CoverageHeatmap({ models }) {
  const [sortBy, setSortBy] = useState('coverage')
  const [showAll, setShowAll] = useState(false)

  const sorted = useMemo(() => {
    return [...models]
      .map(m => ({
        ...m,
        _coverage: ALL_BENCHMARKS.filter(b => m[b.key] != null).length,
      }))
      .filter(m => m._coverage > 0)
      .sort((a, b) => {
        if (sortBy === 'coverage') return b._coverage - a._coverage || (b.libero_avg ?? -1) - (a.libero_avg ?? -1)
        if (sortBy === 'libero_avg') return (b.libero_avg ?? -1) - (a.libero_avg ?? -1)
        if (sortBy === 'name') return a.name.localeCompare(b.name)
        if (sortBy === 'date') return (b.date || '').localeCompare(a.date || '')
        return 0
      })
  }, [models, sortBy])

  const displayed = showAll ? sorted : sorted.slice(0, 30)

  // Coverage statistics
  const stats = useMemo(() => {
    const coverageDist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] // 0-9 benchmarks
    sorted.forEach(m => {
      coverageDist[m._coverage] = (coverageDist[m._coverage] || 0) + 1
    })
    const avgCoverage = sorted.length > 0
      ? (sorted.reduce((s, m) => s + m._coverage, 0) / sorted.length).toFixed(1)
      : 0
    const fullCoverage = sorted.filter(m => m._coverage >= 4).length
    return { coverageDist, avgCoverage, fullCoverage, total: sorted.length }
  }, [sorted])

  // Benchmark popularity
  const popularity = useMemo(() => {
    return ALL_BENCHMARKS.map(b => ({
      ...b,
      count: models.filter(m => m[b.key] != null).length,
    })).sort((a, b) => b.count - a.count)
  }, [models])

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-white mb-1">Benchmark Coverage Heatmap</h3>
        <p className="text-[11px] text-zinc-500">
          Which models are evaluated on which benchmarks — revealing evaluation gaps and over-tested suites
        </p>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Models w/ Data', value: stats.total, color: 'text-white' },
          { label: 'Avg Coverage', value: `${stats.avgCoverage}`, color: 'text-purple-400' },
          { label: '4+ Benchmarks', value: stats.fullCoverage, color: 'text-emerald-400' },
          { label: 'Benchmarks', value: ALL_BENCHMARKS.length, color: 'text-amber-400' },
        ].map(s => (
          <div key={s.label} className="border border-zinc-800 rounded-lg p-2 text-center">
            <div className={`text-base font-bold ${s.color}`}>{s.value}</div>
            <div className="text-[9px] text-zinc-500">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Benchmark Popularity Bar */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
          Benchmark Popularity
        </h4>
        <div className="space-y-1.5">
          {popularity.map(b => (
            <div key={b.key} className="flex items-center gap-2">
              <span className="text-[10px] text-zinc-400 w-16 text-right shrink-0">{b.label}</span>
              <div className="flex-1 h-5 bg-zinc-800 rounded-full overflow-hidden relative">
                <div
                  className="h-full rounded-full flex items-center justify-end pr-2"
                  style={{
                    width: `${Math.max((b.count / models.length) * 100, 3)}%`,
                    backgroundColor: b.count > 15 ? '#7F77DD' : b.count > 5 ? '#3498DB' : '#71717a',
                  }}
                >
                  <span className="text-[9px] text-white font-medium">{b.count}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Coverage Distribution */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
          Models by # Benchmarks Covered
        </h4>
        <div className="flex gap-1 items-end h-20">
          {stats.coverageDist.slice(1).map((count, i) => {
            const benchCount = i + 1
            const maxCount = Math.max(...stats.coverageDist.slice(1))
            const height = maxCount > 0 ? (count / maxCount) * 100 : 0
            return (
              <div key={benchCount} className="flex-1 flex flex-col items-center gap-0.5">
                <span className="text-[9px] text-zinc-500">{count || ''}</span>
                <div
                  className="w-full rounded-t"
                  style={{
                    height: `${Math.max(height, count > 0 ? 4 : 0)}%`,
                    backgroundColor: benchCount >= 4 ? '#10b981' : benchCount >= 2 ? '#3498DB' : '#71717a',
                    minHeight: count > 0 ? 3 : 0,
                  }}
                />
                <span className="text-[9px] text-zinc-600">{benchCount}</span>
              </div>
            )
          })}
        </div>
        <div className="text-center text-[9px] text-zinc-600 mt-1">Number of benchmarks covered</div>
      </div>

      {/* Sort controls */}
      <div className="flex items-center justify-between">
        <div className="flex gap-1.5">
          {SORT_OPTIONS.map(opt => (
            <button
              key={opt.key}
              onClick={() => setSortBy(opt.key)}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                sortBy === opt.key
                  ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
        >
          {showAll ? 'Show top 30' : `Show all ${sorted.length}`}
        </button>
      </div>

      {/* Heatmap Table */}
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr className="bg-zinc-900/50">
                <th className="text-left px-2 py-2 text-zinc-500 font-medium sticky left-0 bg-zinc-900/95 z-10 min-w-[120px]">Model</th>
                <th className="text-center px-1 py-2 text-zinc-500 font-medium w-8">#</th>
                {ALL_BENCHMARKS.map(b => (
                  <th key={b.key} className="text-center px-1.5 py-2 text-zinc-500 font-medium whitespace-nowrap">
                    <div style={{ writingMode: 'vertical-lr', transform: 'rotate(180deg)', fontSize: '9px' }}>
                      {b.label}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayed.map(m => (
                <tr key={m.name} className="border-t border-zinc-800/30 hover:bg-zinc-900/20">
                  <td className="px-2 py-1.5 text-zinc-300 font-medium whitespace-nowrap sticky left-0 bg-zinc-950/95 z-10">
                    <div className="flex items-center gap-1">
                      {m.name}
                      {m.open_source && (
                        <span className="text-[7px] px-1 py-0 rounded bg-emerald-500/10 text-emerald-400">O</span>
                      )}
                    </div>
                  </td>
                  <td className="text-center px-1 py-1.5 tabular-nums text-zinc-500 font-medium">
                    {m._coverage}
                  </td>
                  {ALL_BENCHMARKS.map(b => {
                    const val = m[b.key]
                    const hasData = val != null
                    return (
                      <td
                        key={b.key}
                        className="text-center px-1.5 py-1.5 tabular-nums"
                        style={{ backgroundColor: heatColor(val, b.max) }}
                      >
                        {hasData ? (
                          <span className="text-white font-medium">
                            {b.max === 5 ? val.toFixed(1) : val.toFixed(0)}
                          </span>
                        ) : (
                          <span className="text-zinc-700">·</span>
                        )}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Color legend */}
      <div className="flex items-center gap-3 justify-center text-[9px] text-zinc-500">
        <span>Performance:</span>
        {[
          { label: '90%+', color: 'rgba(16, 185, 129, 0.8)' },
          { label: '75%+', color: 'rgba(16, 185, 129, 0.5)' },
          { label: '60%+', color: 'rgba(250, 204, 21, 0.5)' },
          { label: '40%+', color: 'rgba(249, 115, 22, 0.5)' },
          { label: '<40%', color: 'rgba(239, 68, 68, 0.5)' },
          { label: 'No data', color: 'transparent' },
        ].map(c => (
          <span key={c.label} className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-sm border border-zinc-700" style={{ backgroundColor: c.color }} />
            {c.label}
          </span>
        ))}
      </div>
    </div>
  )
}
