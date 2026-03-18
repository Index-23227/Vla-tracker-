import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const SUITE_LABELS = { libero_spatial: 'Spatial', libero_object: 'Object', libero_goal: 'Goal', libero_long: 'Long' }
const TOP_COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E']

export default function ModelCompare({ models }) {
  const topModels = useMemo(() => {
    return [...models]
      .filter(m => m.benchmarks?.libero)
      .map(m => {
        const lib = m.benchmarks.libero
        const scores = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_long']
          .map(s => lib[s]).filter(v => v != null)
        const avg = scores.length === 4 ? scores.reduce((a, b) => a + b, 0) / 4 : 0
        return { ...m, avg }
      })
      .sort((a, b) => b.avg - a.avg)
      .slice(0, 4)
  }, [models])

  const barData = useMemo(() => {
    return Object.entries(SUITE_LABELS).map(([key, label]) => {
      const point = { suite: label }
      topModels.forEach(m => {
        point[m.name] = m.benchmarks.libero[key]
      })
      return point
    })
  }, [topModels])

  return (
    <div>
      <h3 className="text-sm font-semibold text-white mb-3">Top 4 Models Across LIBERO Suites</h3>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={barData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="suite" tick={{ fontSize: 12, fill: '#a1a1aa' }} />
          <YAxis domain={[40, 100]} tick={{ fontSize: 11, fill: '#71717a' }} />
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
              fill={TOP_COLORS[i]}
              radius={[4, 4, 0, 0]}
              barSize={18}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>

      <div className="flex gap-4 justify-center mt-2 flex-wrap">
        {topModels.map((m, i) => (
          <span key={m.name} className="flex items-center gap-1.5 text-[11px] text-zinc-400">
            <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ backgroundColor: TOP_COLORS[i] }} />
            {m.name}
          </span>
        ))}
      </div>

      <div className="mt-4 p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg text-xs text-amber-200/80">
        <strong>AI Insight:</strong> LIBERO-Long remains the most challenging suite. Even the top model
        ({topModels[0]?.name}) scores {topModels[0]?.benchmarks?.libero?.libero_long}% here vs{' '}
        {topModels[0]?.benchmarks?.libero?.libero_spatial}% on Spatial — a{' '}
        {(topModels[0]?.benchmarks?.libero?.libero_spatial - topModels[0]?.benchmarks?.libero?.libero_long).toFixed(1)}%p gap.
        CoT-VLA's chain-of-thought approach shows the smallest gap, suggesting explicit reasoning helps sequential tasks.
      </div>
    </div>
  )
}
