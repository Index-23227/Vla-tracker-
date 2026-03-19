import { useState, useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ZAxis,
} from 'recharts'

const BENCH_OPTIONS = [
  { key: 'libero_avg', label: 'LIBERO', max: 100, unit: '%' },
  { key: 'calvin_avg', label: 'CALVIN', max: 5, unit: 'avg len' },
  { key: 'robotwin_v1_avg', label: 'RoboTwin v1', max: 100, unit: '%' },
]

const AXIS_OPTIONS = [
  { key: 'params', label: 'Parameters' },
  { key: 'hz', label: 'Inference Speed (Hz)' },
]

function parseParams(paramStr) {
  if (!paramStr || paramStr === 'unknown') return null
  const s = String(paramStr).toLowerCase().replace(/[~≈+]/g, '')
  const match = s.match(/([\d.]+)\s*(b|m|k)?/)
  if (!match) return null
  const num = parseFloat(match[1])
  const unit = match[2]
  if (unit === 'b') return num * 1000
  if (unit === 'k') return num / 1000
  return num
}

function formatParams(v) {
  if (v >= 1000) return `${(v / 1000).toFixed(1)}B`
  return `${v}M`
}

// Compute Pareto frontier (higher Y and lower X is better for params, higher X for hz)
function computePareto(data, xKey, higherXBetter) {
  if (data.length === 0) return []
  const sorted = [...data].sort((a, b) => higherXBetter ? b[xKey] - a[xKey] : a[xKey] - b[xKey])
  const frontier = [sorted[0]]
  let bestY = sorted[0].score
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].score > bestY) {
      frontier.push(sorted[i])
      bestY = sorted[i].score
    }
  }
  return frontier.map(d => d.name)
}

export default function EfficiencyRanking({ models }) {
  const [benchKey, setBenchKey] = useState('libero_avg')
  const [axisKey, setAxisKey] = useState('params')

  const benchCfg = BENCH_OPTIONS.find(b => b.key === benchKey)

  const data = useMemo(() => {
    return models
      .filter(m => {
        const score = m[benchKey]
        if (score == null) return false
        if (axisKey === 'params') return parseParams(m.architecture?.parameters) != null
        if (axisKey === 'hz') return m.inference_hz != null
        return false
      })
      .map(m => ({
        name: m.name,
        score: m[benchKey],
        params: parseParams(m.architecture?.parameters),
        hz: m.inference_hz,
        oss: m.open_source,
        x: axisKey === 'params' ? parseParams(m.architecture?.parameters) : m.inference_hz,
      }))
  }, [models, benchKey, axisKey])

  const paretoNames = useMemo(() => {
    return computePareto(data, 'x', axisKey === 'hz')
  }, [data, axisKey])

  // Efficiency score: performance / log(params) or performance * log(hz)
  const ranking = useMemo(() => {
    return data
      .map(d => {
        let efficiency
        if (axisKey === 'params') {
          efficiency = d.score / Math.log2(d.x + 1)
        } else {
          efficiency = d.score * Math.log2(d.x + 1)
        }
        return { ...d, efficiency, isPareto: paretoNames.includes(d.name) }
      })
      .sort((a, b) => b.efficiency - a.efficiency)
  }, [data, axisKey, paretoNames])

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-white mb-1">Efficiency Ranking</h3>
        <p className="text-[11px] text-zinc-500">
          Pareto frontier analysis — finding models with the best performance-to-cost tradeoff
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div>
          <div className="text-[10px] text-zinc-500 mb-1 uppercase tracking-wider">Benchmark</div>
          <div className="flex gap-1.5">
            {BENCH_OPTIONS.map(b => (
              <button
                key={b.key}
                onClick={() => setBenchKey(b.key)}
                className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                  benchKey === b.key
                    ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                    : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
                }`}
              >
                {b.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="text-[10px] text-zinc-500 mb-1 uppercase tracking-wider">X-Axis</div>
          <div className="flex gap-1.5">
            {AXIS_OPTIONS.map(a => (
              <button
                key={a.key}
                onClick={() => setAxisKey(a.key)}
                className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                  axisKey === a.key
                    ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-300 font-semibold'
                    : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
                }`}
              >
                {a.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Scatter Plot with Pareto */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
          {benchCfg.label} vs {axisKey === 'params' ? 'Parameters' : 'Inference Speed'}
        </h4>
        <p className="text-[10px] text-zinc-600 mb-3">
          {axisKey === 'params'
            ? 'Stars = Pareto optimal (best performance for given size or smaller)'
            : 'Stars = Pareto optimal (best performance for given speed or faster)'}
        </p>
        {data.length > 0 ? (
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="x"
                name={axisKey === 'params' ? 'Parameters' : 'Hz'}
                type="number"
                tick={{ fontSize: 10, fill: '#71717a' }}
                tickFormatter={v => axisKey === 'params' ? formatParams(v) : `${v}Hz`}
                label={{ value: axisKey === 'params' ? 'Parameters' : 'Inference Speed (Hz)', position: 'bottom', fontSize: 10, fill: '#52525b', offset: 5 }}
              />
              <YAxis
                dataKey="score"
                name={benchCfg.label}
                type="number"
                tick={{ fontSize: 10, fill: '#71717a' }}
                label={{ value: `${benchCfg.label} (${benchCfg.unit})`, angle: -90, position: 'insideLeft', fontSize: 10, fill: '#52525b' }}
              />
              <ZAxis range={[80, 200]} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload || !payload.length) return null
                  const d = payload[0].payload
                  return (
                    <div style={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5', padding: '8px 10px' }}>
                      <div style={{ fontWeight: 'bold', marginBottom: 3 }}>
                        {d.name} {paretoNames.includes(d.name) ? '★' : ''}
                      </div>
                      <div>{benchCfg.label}: {benchCfg.max === 5 ? d.score.toFixed(2) : d.score.toFixed(1)}{benchCfg.unit === '%' ? '%' : ` ${benchCfg.unit}`}</div>
                      {d.params && <div>Params: {formatParams(d.params)}</div>}
                      {d.hz && <div>Speed: {d.hz} Hz</div>}
                      <div>{d.oss ? 'Open Source' : 'Closed'}</div>
                      {paretoNames.includes(d.name) && <div style={{ color: '#f59e0b', marginTop: 2 }}>Pareto Optimal</div>}
                    </div>
                  )
                }}
              />
              <Scatter data={data}>
                {data.map((entry, i) => (
                  <Cell
                    key={entry.name}
                    fill={paretoNames.includes(entry.name) ? '#f59e0b' : entry.oss ? '#1D9E75' : '#7F77DD'}
                    fillOpacity={paretoNames.includes(entry.name) ? 1 : 0.7}
                    stroke={paretoNames.includes(entry.name) ? '#f59e0b' : 'none'}
                    strokeWidth={paretoNames.includes(entry.name) ? 2 : 0}
                    r={paretoNames.includes(entry.name) ? 7 : 5}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center py-12 text-zinc-500 text-xs">Not enough data for this combination</div>
        )}
        <div className="flex gap-4 justify-center mt-1">
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#f59e0b' }} /> Pareto Optimal
          </span>
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#1D9E75' }} /> Open Source
          </span>
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: '#7F77DD' }} /> Closed
          </span>
        </div>
      </div>

      {/* Efficiency Ranking Table */}
      <div className="border border-zinc-800 rounded-xl overflow-hidden">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider px-4 pt-3 pb-2">
          Efficiency Ranking
          <span className="text-[10px] text-zinc-600 font-normal ml-2">
            ({axisKey === 'params' ? 'score / log₂(params)' : 'score × log₂(hz)'})
          </span>
        </h4>
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-zinc-900/50">
              <th className="px-3 py-2 text-left text-[10px] font-semibold text-zinc-500 uppercase w-8">#</th>
              <th className="px-3 py-2 text-left text-[10px] font-semibold text-zinc-500 uppercase">Model</th>
              <th className="px-3 py-2 text-right text-[10px] font-semibold text-zinc-500 uppercase">{benchCfg.label}</th>
              <th className="px-3 py-2 text-right text-[10px] font-semibold text-zinc-500 uppercase">
                {axisKey === 'params' ? 'Params' : 'Speed'}
              </th>
              <th className="px-3 py-2 text-right text-[10px] font-semibold text-zinc-500 uppercase">Efficiency</th>
            </tr>
          </thead>
          <tbody>
            {ranking.slice(0, 15).map((d, i) => (
              <tr key={d.name} className="border-t border-zinc-800/50 hover:bg-zinc-900/30">
                <td className="px-3 py-2 font-medium">
                  {d.isPareto ? <span className="text-amber-400">★</span> : <span className="text-zinc-500">{i + 1}</span>}
                </td>
                <td className="px-3 py-2">
                  <span className="font-semibold text-white">{d.name}</span>
                  {d.oss && <span className="ml-1.5 text-[9px] px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-400">OSS</span>}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-zinc-300">
                  {benchCfg.max === 5 ? d.score.toFixed(2) : d.score.toFixed(1)}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-zinc-300">
                  {axisKey === 'params' ? formatParams(d.x) : `${d.x} Hz`}
                </td>
                <td className={`px-3 py-2 text-right tabular-nums font-bold ${i === 0 ? 'text-emerald-400' : 'text-white'}`}>
                  {d.efficiency.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="px-3 py-2 bg-zinc-900/50 border border-zinc-800/50 rounded-lg text-[10px] text-zinc-500">
        <strong className="text-zinc-400">How to read:</strong> Pareto optimal models (★) achieve the best performance
        for their {axisKey === 'params' ? 'parameter count' : 'inference speed'} — no other model is both
        {axisKey === 'params' ? ' smaller and better performing' : ' faster and better performing'}.
        The efficiency score rewards models that achieve high benchmark scores with {axisKey === 'params' ? 'fewer parameters' : 'faster inference'}.
      </div>
    </div>
  )
}
