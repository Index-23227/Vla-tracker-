import { useState, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function PerformanceChart({ modelHistory }) {
  const allModels = modelHistory.map(m => m.name)
  const [selected, setSelected] = useState(allModels.slice(0, 4))

  const chartData = useMemo(() => {
    const months = ['Oct 24', 'Jan 25', 'Apr 25', 'Jul 25', 'Oct 25', 'Jan 26']
    return months.map(month => {
      const point = { month }
      modelHistory
        .filter(m => selected.includes(m.name))
        .forEach(m => {
          const h = m.history.find(h => h.month === month)
          if (h && h.avg != null) point[m.name] = h.avg
        })
      return point
    })
  }, [selected, modelHistory])

  const toggle = (name) => {
    setSelected(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    )
  }

  return (
    <div>
      <h3 className="text-sm font-semibold text-white mb-1">LIBERO Average Score Over Time</h3>
      <div className="flex gap-3 mb-4 flex-wrap">
        {modelHistory.map(m => (
          <label key={m.name} className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={selected.includes(m.name)}
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
          <YAxis domain={[60, 100]} tick={{ fontSize: 11, fill: '#71717a' }} />
          <Tooltip
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: '1px solid #3f3f46',
              backgroundColor: '#18181b',
              color: '#e5e5e5',
            }}
          />
          {modelHistory
            .filter(m => selected.includes(m.name))
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
        Models appear when first published. Gap = not yet released.
      </p>
    </div>
  )
}
