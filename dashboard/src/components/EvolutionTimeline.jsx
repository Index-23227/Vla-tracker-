import { useMemo, useState } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, CartesianGrid, Cell,
} from 'recharts'
import { ACTION_HEAD_COLORS, ACTION_HEAD_LABELS } from '../constants/benchmarks'

const CATEGORIES = Object.keys(ACTION_HEAD_COLORS)

function toQuarter(dateStr) {
  if (!dateStr) return null
  const [y, m] = dateStr.split('-').map(Number)
  return `${y}-Q${Math.ceil(m / 3)}`
}

function toMonth(dateStr) {
  return dateStr ? dateStr.slice(0, 7) : null
}

export default function EvolutionTimeline({ models }) {
  const [viewMode, setViewMode] = useState('quarter')

  // 1. Cumulative stacked area
  const areaData = useMemo(() => {
    const groupFn = viewMode === 'quarter' ? toQuarter : toMonth
    const sorted = [...models].filter(m => m.date).sort((a, b) => a.date.localeCompare(b.date))
    const periods = [...new Set(sorted.map(m => groupFn(m.date)).filter(Boolean))].sort()
    const cumulative = {}
    CATEGORIES.forEach(c => { cumulative[c] = 0 })
    return periods.map(period => {
      sorted.forEach(m => {
        if (groupFn(m.date) === period) {
          const cat = m.architecture?.action_head_category || 'other'
          cumulative[cat] = (cumulative[cat] || 0) + 1
        }
      })
      return { period, ...{ ...cumulative } }
    })
  }, [models, viewMode])

  // 2. Scatter: date vs LIBERO
  const scatterData = useMemo(() => {
    return models
      .filter(m => m.date && m.libero_avg)
      .map(m => ({
        date: new Date(m.date).getTime(),
        score: m.libero_avg,
        name: m.name,
        category: m.architecture?.action_head_category || 'other',
      }))
      .sort((a, b) => a.date - b.date)
  }, [models])

  // 3. Milestones
  const milestones = useMemo(() => {
    const ms = {}
    ;[...models].filter(m => m.date).sort((a, b) => a.date.localeCompare(b.date)).forEach(m => {
      const cat = m.architecture?.action_head_category || 'other'
      if (!ms[cat]) ms[cat] = { first: m, best: m, count: 0 }
      ms[cat].count++
      if (m.libero_avg && (!ms[cat].best.libero_avg || m.libero_avg > ms[cat].best.libero_avg)) {
        ms[cat].best = m
      }
    })
    return ms
  }, [models])

  // 4. New per period
  const newPerPeriod = useMemo(() => {
    const groupFn = viewMode === 'quarter' ? toQuarter : toMonth
    const sorted = [...models].filter(m => m.date).sort((a, b) => a.date.localeCompare(b.date))
    const periods = [...new Set(sorted.map(m => groupFn(m.date)).filter(Boolean))].sort()
    return periods.map(period => {
      const counts = {}
      CATEGORIES.forEach(c => { counts[c] = 0 })
      sorted.forEach(m => {
        if (groupFn(m.date) === period) {
          counts[m.architecture?.action_head_category || 'other'] = (counts[m.architecture?.action_head_category || 'other'] || 0) + 1
        }
      })
      return { period, ...counts }
    })
  }, [models, viewMode])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-white">Evolution Timeline</h2>
          <p className="text-xs text-zinc-500 mt-0.5">How VLA action generation paradigms evolved over time</p>
        </div>
        <div className="flex gap-1">
          {[{ id: 'quarter', label: 'Quarterly' }, { id: 'month', label: 'Monthly' }].map(v => (
            <button key={v.id} onClick={() => setViewMode(v.id)}
              className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${viewMode === v.id ? 'border-zinc-600 bg-zinc-700 text-white' : 'border-zinc-800 text-zinc-500'}`}>
              {v.label}
            </button>
          ))}
        </div>
      </div>

      {/* Paradigm Shift */}
      <div className="bg-zinc-800/30 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Paradigm Shift: Cumulative Models by Action Head</h3>
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={areaData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="period" tick={{ fontSize: 10, fill: '#888' }} />
            <YAxis tick={{ fontSize: 10, fill: '#888' }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', borderRadius: 8, fontSize: 11 }} />
            {CATEGORIES.filter(c => c !== 'other').map(cat => (
              <Area key={cat} type="monotone" dataKey={cat} stackId="1"
                stroke={ACTION_HEAD_COLORS[cat]} fill={ACTION_HEAD_COLORS[cat]}
                fillOpacity={0.6} name={ACTION_HEAD_LABELS?.[cat] || cat} />
            ))}
            <Area type="monotone" dataKey="other" stackId="1" stroke="#71717a" fill="#71717a" fillOpacity={0.3} name="Other" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* LIBERO vs Time */}
      <div className="bg-zinc-800/30 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">LIBERO Performance Over Time</h3>
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" type="number" domain={['dataMin', 'dataMax']}
              tickFormatter={ts => new Date(ts).toLocaleDateString('en', { month: 'short', year: '2-digit' })}
              tick={{ fontSize: 10, fill: '#888' }} />
            <YAxis dataKey="score" domain={[60, 100]} tick={{ fontSize: 10, fill: '#888' }} />
            <Tooltip content={({ payload }) => {
              if (!payload?.[0]) return null
              const d = payload[0].payload
              return (
                <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-2 text-xs">
                  <div className="font-bold text-white">{d.name}</div>
                  <div className="text-zinc-400">{new Date(d.date).toLocaleDateString('en', { month: 'short', year: 'numeric' })}</div>
                  <div style={{ color: ACTION_HEAD_COLORS[d.category] }}>{ACTION_HEAD_LABELS?.[d.category] || d.category}</div>
                  <div className="text-emerald-400 font-mono">{d.score.toFixed(1)}%</div>
                </div>
              )
            }} />
            <Scatter data={scatterData}>
              {scatterData.map((entry, i) => (
                <Cell key={i} fill={ACTION_HEAD_COLORS[entry.category] || '#71717a'} fillOpacity={0.8} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <div className="flex flex-wrap gap-3 mt-2 justify-center">
          {CATEGORIES.filter(c => c !== 'other').map(cat => (
            <div key={cat} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: ACTION_HEAD_COLORS[cat] }} />
              <span className="text-[10px] text-zinc-500">{ACTION_HEAD_LABELS?.[cat] || cat}</span>
            </div>
          ))}
        </div>
      </div>

      {/* New Models Per Period */}
      <div className="bg-zinc-800/30 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">New Models Per {viewMode === 'quarter' ? 'Quarter' : 'Month'}</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={newPerPeriod}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="period" tick={{ fontSize: 10, fill: '#888' }} />
            <YAxis tick={{ fontSize: 10, fill: '#888' }} />
            <Tooltip contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', borderRadius: 8, fontSize: 11 }} />
            {CATEGORIES.filter(c => c !== 'other').map(cat => (
              <Area key={cat} type="monotone" dataKey={cat} stackId="1"
                stroke={ACTION_HEAD_COLORS[cat]} fill={ACTION_HEAD_COLORS[cat]}
                fillOpacity={0.7} name={ACTION_HEAD_LABELS?.[cat] || cat} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Milestones */}
      <div className="bg-zinc-800/30 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-zinc-300 mb-3">Category Milestones</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-700">
                <th className="text-left px-2 py-1.5 text-zinc-500">Category</th>
                <th className="text-left px-2 py-1.5 text-zinc-500">Count</th>
                <th className="text-left px-2 py-1.5 text-zinc-500">First Model</th>
                <th className="text-left px-2 py-1.5 text-zinc-500">Date</th>
                <th className="text-left px-2 py-1.5 text-zinc-500">Best LIBERO</th>
                <th className="text-left px-2 py-1.5 text-zinc-500">Best Model</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(milestones)
                .filter(([cat]) => cat !== 'other')
                .sort((a, b) => b[1].count - a[1].count)
                .map(([cat, ms]) => (
                  <tr key={cat} className="border-b border-zinc-800/50">
                    <td className="px-2 py-1.5">
                      <span className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full" style={{ backgroundColor: ACTION_HEAD_COLORS[cat] }} />
                        <span className="text-zinc-300">{ACTION_HEAD_LABELS?.[cat] || cat}</span>
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-zinc-400 font-mono">{ms.count}</td>
                    <td className="px-2 py-1.5 text-zinc-300">{ms.first.name}</td>
                    <td className="px-2 py-1.5 text-zinc-500">{ms.first.date}</td>
                    <td className="px-2 py-1.5 text-emerald-400 font-mono">{ms.best.libero_avg?.toFixed(1) || '—'}</td>
                    <td className="px-2 py-1.5 text-zinc-300">{ms.best.libero_avg ? ms.best.name : '—'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
