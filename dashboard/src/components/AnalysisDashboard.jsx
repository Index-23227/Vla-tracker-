import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell,
  PieChart, Pie,
} from 'recharts'
import { BENCHMARK_LIST, ACTION_HEAD_COLORS, COLORS, heatColor } from '../constants/benchmarks'

const PIE_COLORS = COLORS

function parseParams(paramStr) {
  if (!paramStr || paramStr === 'unknown') return null
  const s = String(paramStr).toLowerCase().replace(/[~≈]/g, '')
  const match = s.match(/([\d.]+)\s*(b|m|k)?/)
  if (!match) return null
  const num = parseFloat(match[1])
  const unit = match[2]
  if (unit === 'b') return num * 1000
  if (unit === 'k') return num / 1000
  return num // assume M
}

const BENCH_KEYS = BENCHMARK_LIST

export default function AnalysisDashboard({ models }) {
  // --- Architecture Distribution ---
  const actionHeadDist = useMemo(() => {
    const counts = {}
    models.forEach(m => {
      const head = m.architecture?.action_head || 'unknown'
      // Normalize
      const key = Object.keys(ACTION_HEAD_COLORS).find(k => head.toLowerCase().includes(k)) || 'other'
      counts[key] = (counts[key] || 0) + 1
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value }))
  }, [models])

  // --- Open-source vs Closed ---
  const ossStats = useMemo(() => {
    const oss = models.filter(m => m.open_source)
    const closed = models.filter(m => !m.open_source)
    const ossWithLibero = oss.filter(m => m.libero_avg != null)
    const closedWithLibero = closed.filter(m => m.libero_avg != null)
    const ossAvg = ossWithLibero.length > 0
      ? ossWithLibero.reduce((s, m) => s + m.libero_avg, 0) / ossWithLibero.length : 0
    const closedAvg = closedWithLibero.length > 0
      ? closedWithLibero.reduce((s, m) => s + m.libero_avg, 0) / closedWithLibero.length : 0
    return {
      oss: oss.length,
      closed: closed.length,
      ossAvg: ossAvg.toFixed(1),
      closedAvg: closedAvg.toFixed(1),
      ossTop: ossWithLibero.sort((a, b) => b.libero_avg - a.libero_avg)[0],
      closedTop: closedWithLibero.sort((a, b) => b.libero_avg - a.libero_avg)[0],
    }
  }, [models])

  // --- Parameter vs Performance scatter ---
  const scatterData = useMemo(() => {
    return models
      .filter(m => m.libero_avg != null && parseParams(m.architecture?.parameters) != null)
      .map(m => ({
        name: m.name,
        params: parseParams(m.architecture?.parameters),
        score: m.libero_avg,
        oss: m.open_source,
        actionHead: m.architecture?.action_head || 'unknown',
      }))
  }, [models])

  // --- Timeline: models per quarter ---
  const timelineData = useMemo(() => {
    const quarters = {}
    models.forEach(m => {
      if (!m.date) return
      const d = new Date(m.date)
      const q = `${d.getFullYear()} Q${Math.floor(d.getMonth() / 3) + 1}`
      if (!quarters[q]) quarters[q] = { quarter: q, count: 0, models: [] }
      quarters[q].count++
      quarters[q].models.push(m.name)
    })
    return Object.values(quarters).sort((a, b) => a.quarter.localeCompare(b.quarter))
  }, [models])

  // --- Benchmark Coverage ---
  const benchCoverage = useMemo(() => {
    return BENCH_KEYS.map(b => ({
      name: b.label,
      count: models.filter(m => m[b.key] != null).length,
    }))
  }, [models])

  // --- Benchmark Heatmap data ---
  const heatmapModels = useMemo(() => {
    return [...models]
      .filter(m => BENCH_KEYS.some(b => m[b.key] != null))
      .sort((a, b) => (b.libero_avg ?? -1) - (a.libero_avg ?? -1))
      .slice(0, Math.min(40, models.length))
  }, [models])

  // --- Action Head vs Performance ---
  const actionHeadPerf = useMemo(() => {
    const groups = {}
    models.forEach(m => {
      if (m.libero_avg == null) return
      const head = m.architecture?.action_head || 'unknown'
      const key = Object.keys(ACTION_HEAD_COLORS).find(k => head.toLowerCase().includes(k)) || 'other'
      if (!groups[key]) groups[key] = { scores: [], count: 0 }
      groups[key].scores.push(m.libero_avg)
      groups[key].count++
    })
    return Object.entries(groups)
      .map(([name, g]) => ({
        name,
        avg: +(g.scores.reduce((a, b) => a + b, 0) / g.scores.length).toFixed(1),
        best: +Math.max(...g.scores).toFixed(1),
        count: g.count,
      }))
      .sort((a, b) => b.avg - a.avg)
  }, [models])

  // --- Inference Speed distribution ---
  const speedDist = useMemo(() => {
    const buckets = [
      { label: '<5 Hz', min: 0, max: 5, count: 0, models: [] },
      { label: '5-10 Hz', min: 5, max: 10, count: 0, models: [] },
      { label: '10-30 Hz', min: 10, max: 30, count: 0, models: [] },
      { label: '30-60 Hz', min: 30, max: 60, count: 0, models: [] },
      { label: '60+ Hz', min: 60, max: Infinity, count: 0, models: [] },
    ]
    models.forEach(m => {
      if (m.inference_hz == null) return
      const bucket = buckets.find(b => m.inference_hz >= b.min && m.inference_hz < b.max)
      if (bucket) { bucket.count++; bucket.models.push(m.name) }
    })
    return buckets
  }, [models])

  // --- VLM Backbone Distribution ---
  const backboneDist = useMemo(() => {
    const counts = {}
    models.forEach(m => {
      const bb = m.architecture?.backbone || ''
      if (!bb) { counts['unknown'] = (counts['unknown'] || 0) + 1; return }
      // Normalize to family names
      const lower = bb.toLowerCase()
      let key
      if (lower.includes('prismatic') || (lower.includes('siglip') && lower.includes('dino'))) key = 'PrismaticVLM'
      else if (lower.includes('paligemma') || (lower.includes('siglip') && lower.includes('gemma'))) key = 'PaliGemma'
      else if (lower.includes('gemma') && !lower.includes('siglip')) key = 'PaliGemma'
      else if (lower.includes('internvl') || lower.includes('internvit')) key = 'InternVL'
      else if (lower.includes('qwen')) key = 'Qwen-VL'
      else if (lower.includes('eagle') || lower.includes('cosmos')) key = 'NVIDIA VLM'
      else if (lower.includes('florence')) key = 'Florence'
      else if (lower.includes('siglip') && !lower.includes('dino') && !lower.includes('gemma')) key = 'SigLIP'
      else if (lower.includes('clip') || lower.includes('show-o')) key = 'CLIP'
      else if (lower.includes('blip')) key = 'BLIP'
      else if (lower.includes('diffusion') || lower.includes('video')) key = 'Video/Diffusion'
      else key = 'Other'
      counts[key] = (counts[key] || 0) + 1
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value }))
  }, [models])

  const BACKBONE_COLORS = {
    'PrismaticVLM': '#8b5cf6', 'PaliGemma': '#06b6d4', 'InternVL': '#f59e0b',
    'Qwen-VL': '#ef4444', 'NVIDIA VLM': '#22c55e', 'SigLIP': '#ec4899',
    'CLIP': '#3b82f6', 'Florence': '#a855f7', 'BLIP': '#14b8a6',
    'Video/Diffusion': '#f97316', 'Other': '#6b7280', 'unknown': '#404040',
  }

  // --- Eval Condition Distribution ---
  const evalCondDist = useMemo(() => {
    const counts = { 'Fine-tuned': 0, 'RL-trained': 0, 'Zero-shot': 0, 'Unknown': 0 }
    models.forEach(m => {
      const conds = m.eval_conditions || {}
      const vals = Object.values(conds)
      if (vals.length === 0) { counts['Unknown']++; return }
      const hasRL = vals.some(v => /rl|grpo|reinforcement/i.test(v))
      const hasZS = vals.some(v => /zero.?shot/i.test(v))
      if (hasRL) counts['RL-trained']++
      else if (hasZS) counts['Zero-shot']++
      else counts['Fine-tuned']++
    })
    return Object.entries(counts)
      .filter(([, v]) => v > 0)
      .map(([name, value]) => ({ name, value }))
  }, [models])

  const EVAL_COND_COLORS = {
    'Fine-tuned': '#3b82f6', 'RL-trained': '#f43f5e', 'Zero-shot': '#f59e0b', 'Unknown': '#404040',
  }

  // --- Venue Distribution ---
  const venueDist = useMemo(() => {
    const counts = {}
    models.forEach(m => {
      const v = m.venue || 'preprint'
      counts[v] = (counts[v] || 0) + 1
    })
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, value]) => ({ name, value }))
  }, [models])

  // --- Key stats ---
  const stats = useMemo(() => {
    const withLibero = models.filter(m => m.libero_avg != null)
    const avgScore = withLibero.length > 0
      ? (withLibero.reduce((s, m) => s + m.libero_avg, 0) / withLibero.length).toFixed(1) : '—'
    const latestModel = [...models].sort((a, b) => (b.date || '').localeCompare(a.date || ''))[0]
    const withHz = models.filter(m => m.inference_hz != null)
    const avgHz = withHz.length > 0
      ? (withHz.reduce((s, m) => s + m.inference_hz, 0) / withHz.length).toFixed(0) : '—'
    return {
      total: models.length,
      avgScore,
      latestModel: latestModel?.name,
      latestDate: latestModel?.date,
      avgHz,
      ossPercent: ((models.filter(m => m.open_source).length / models.length) * 100).toFixed(0),
    }
  }, [models])

  return (
    <div className="space-y-6">
      {/* Key Stats Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Total Models', value: stats.total, color: 'text-white' },
          { label: 'Avg LIBERO', value: stats.avgScore, color: 'text-purple-400' },
          { label: 'Open Source', value: `${stats.ossPercent}%`, color: 'text-emerald-400' },
          { label: 'Avg Speed', value: `${stats.avgHz} Hz`, color: 'text-amber-400' },
        ].map(s => (
          <div key={s.label} className="border border-zinc-800 rounded-xl p-3 text-center">
            <div className={`text-xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-[10px] text-zinc-500 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Row 1: Action Head + Venue */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Action Head Distribution */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Action Head Distribution
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={actionHeadDist}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={70}
                innerRadius={35}
                strokeWidth={0}
                label={({ name, value }) => `${name} (${value})`}
                labelLine={false}
              >
                {actionHeadDist.map((entry, i) => (
                  <Cell key={entry.name} fill={ACTION_HEAD_COLORS[entry.name] || PIE_COLORS[i % PIE_COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Benchmark Coverage */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Benchmark Adoption
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={benchCoverage} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: '#71717a' }} />
              <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: '#a1a1aa' }} width={70} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
              />
              <Bar dataKey="count" fill="#7F77DD" radius={[0, 4, 4, 0]} barSize={14} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 1b: VLM Backbone + Eval Condition */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* VLM Backbone Distribution */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            VLM Backbone Distribution
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={backboneDist}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={70}
                innerRadius={35}
                strokeWidth={0}
                label={({ name, value }) => `${name} (${value})`}
                labelLine={false}
              >
                {backboneDist.map((entry) => (
                  <Cell key={entry.name} fill={BACKBONE_COLORS[entry.name] || '#6b7280'} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Eval Condition Distribution */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Evaluation Condition Distribution
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <PieChart>
              <Pie
                data={evalCondDist}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={70}
                innerRadius={35}
                strokeWidth={0}
                label={({ name, value }) => `${name} (${value})`}
                labelLine={false}
              >
                {evalCondDist.map((entry) => (
                  <Cell key={entry.name} fill={EVAL_COND_COLORS[entry.name] || '#6b7280'} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Row 2: Parameter vs Performance */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
          Parameters vs LIBERO Performance
        </h4>
        <p className="text-[10px] text-zinc-600 mb-3">Bubble size = parameter count · Green = open-source</p>
        {scatterData.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="params"
                name="Parameters (M)"
                type="number"
                tick={{ fontSize: 10, fill: '#71717a' }}
                tickFormatter={v => v >= 1000 ? `${(v/1000).toFixed(0)}B` : `${v}M`}
                label={{ value: 'Parameters', position: 'bottom', fontSize: 10, fill: '#52525b', offset: 0 }}
              />
              <YAxis
                dataKey="score"
                name="LIBERO Avg"
                type="number"
                tick={{ fontSize: 10, fill: '#71717a' }}
                domain={[40, 100]}
                label={{ value: 'LIBERO Avg (%)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#52525b' }}
              />
              <ZAxis dataKey="params" range={[60, 300]} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
                content={({ payload }) => {
                  if (!payload || !payload.length) return null
                  const d = payload[0].payload
                  return (
                    <div style={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5', padding: '8px 10px' }}>
                      <div style={{ fontWeight: 'bold', marginBottom: 3 }}>{d.name}</div>
                      <div>Parameters: {d.params >= 1000 ? `${(d.params/1000).toFixed(1)}B` : `${d.params}M`}</div>
                      <div>LIBERO: {d.score.toFixed(1)}%</div>
                      <div>{d.oss ? '✓ Open Source' : '✗ Closed'}</div>
                    </div>
                  )
                }}
              />
              <Scatter data={scatterData}>
                {scatterData.map((entry, i) => (
                  <Cell key={i} fill={entry.oss ? '#1D9E75' : '#D85A30'} fillOpacity={0.8} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="text-center py-8 text-zinc-500 text-xs">Not enough data with parameter counts</div>
        )}
        <div className="flex gap-3 justify-center mt-1">
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: '#1D9E75' }} /> Open Source
          </span>
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: '#D85A30' }} /> Closed
          </span>
        </div>
      </div>

      {/* Row 3: Timeline + Open Source */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Publication Timeline */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Publication Timeline
          </h4>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={timelineData} margin={{ top: 0, right: 10, left: -15, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="quarter" tick={{ fontSize: 9, fill: '#a1a1aa' }} />
              <YAxis tick={{ fontSize: 10, fill: '#71717a' }} allowDecimals={false} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
                formatter={(value, name, props) => [
                  `${value} models`,
                  props.payload.models?.join(', ')
                ]}
              />
              <Bar dataKey="count" fill="#3498DB" radius={[4, 4, 0, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Open Source Analysis */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Open Source vs Closed
          </h4>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <div>
                <div className="text-xs text-zinc-400">Open Source</div>
                <div className="text-lg font-bold text-emerald-400">{ossStats.oss} models</div>
                <div className="text-[10px] text-zinc-500">Avg LIBERO: {ossStats.ossAvg}%</div>
              </div>
              <div className="text-right">
                <div className="text-xs text-zinc-400">Closed</div>
                <div className="text-lg font-bold text-orange-400">{ossStats.closed} models</div>
                <div className="text-[10px] text-zinc-500">Avg LIBERO: {ossStats.closedAvg}%</div>
              </div>
            </div>
            {/* Progress bar */}
            <div className="w-full h-3 rounded-full overflow-hidden bg-zinc-800 flex">
              <div
                className="h-full bg-emerald-500/70 rounded-l-full"
                style={{ width: `${(ossStats.oss / (ossStats.oss + ossStats.closed)) * 100}%` }}
              />
              <div
                className="h-full bg-orange-500/70 rounded-r-full"
                style={{ width: `${(ossStats.closed / (ossStats.oss + ossStats.closed)) * 100}%` }}
              />
            </div>
            <div className="text-[10px] text-zinc-500">
              Best OSS: <span className="text-emerald-400">{ossStats.ossTop?.name}</span> ({ossStats.ossTop?.libero_avg}%)
              {ossStats.closedTop && (
                <> · Best Closed: <span className="text-orange-400">{ossStats.closedTop?.name}</span> ({ossStats.closedTop?.libero_avg}%)</>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Benchmark Heatmap */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
          Benchmark Coverage Heatmap
        </h4>
        <p className="text-[10px] text-zinc-600 mb-3">Top {heatmapModels.length} models by LIBERO score · Color intensity = normalized performance</p>
        <div className="overflow-x-auto">
          <table className="w-full text-[10px]">
            <thead>
              <tr>
                <th className="text-left px-2 py-1.5 text-zinc-500 font-medium sticky left-0 bg-zinc-950 z-10">Model</th>
                {BENCH_KEYS.map(b => (
                  <th key={b.key} className="text-center px-2 py-1.5 text-zinc-500 font-medium whitespace-nowrap">{b.label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {heatmapModels.map(m => (
                <tr key={m.name} className="border-t border-zinc-800/30">
                  <td className="px-2 py-1.5 text-zinc-300 font-medium whitespace-nowrap sticky left-0 bg-zinc-950 z-10">
                    <div className="flex items-center gap-1">
                      {m.name}
                      {m.paper_url && (
                        <a href={m.paper_url} target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">↗</a>
                      )}
                    </div>
                  </td>
                  {BENCH_KEYS.map(b => {
                    const val = m[b.key]
                    return (
                      <td key={b.key} className="text-center px-2 py-1.5 tabular-nums" style={{ backgroundColor: heatColor(val, b.max) }}>
                        {val != null ? (b.max === 5 ? val.toFixed(1) : val.toFixed(1)) : '—'}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Row: Action Head Performance + Speed Distribution */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Action Head vs LIBERO Performance */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Action Head vs LIBERO Score
          </h4>
          <div className="space-y-2">
            {actionHeadPerf.map(a => (
              <div key={a.name}>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-zinc-300 font-medium">{a.name} <span className="text-zinc-600">({a.count})</span></span>
                  <span className="text-zinc-400">avg {a.avg} · best {a.best}</span>
                </div>
                <div className="w-full h-4 bg-zinc-800 rounded-full overflow-hidden relative">
                  <div
                    className="h-full rounded-full opacity-40"
                    style={{
                      width: `${a.best}%`,
                      backgroundColor: ACTION_HEAD_COLORS[a.name] || '#71717a',
                    }}
                  />
                  <div
                    className="h-full rounded-full absolute top-0 left-0"
                    style={{
                      width: `${a.avg}%`,
                      backgroundColor: ACTION_HEAD_COLORS[a.name] || '#71717a',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
          <p className="text-[10px] text-zinc-600 mt-2">Solid = average · Faded = best score</p>
        </div>

        {/* Inference Speed Distribution */}
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
            Inference Speed Distribution
          </h4>
          <div className="space-y-2">
            {speedDist.map(b => (
              <div key={b.label}>
                <div className="flex justify-between text-[10px] mb-0.5">
                  <span className="text-zinc-300 font-medium">{b.label}</span>
                  <span className="text-zinc-400">{b.count} models</span>
                </div>
                <div className="w-full h-4 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${Math.max((b.count / models.length) * 100 * 3, b.count > 0 ? 4 : 0)}%`,
                      backgroundColor: b.min >= 30 ? '#10b981' : b.min >= 10 ? '#f59e0b' : '#ef4444',
                    }}
                  />
                </div>
                {b.count > 0 && (
                  <div className="text-[9px] text-zinc-600 mt-0.5 truncate">{b.models.join(', ')}</div>
                )}
              </div>
            ))}
          </div>
          <p className="text-[10px] text-zinc-600 mt-2">
            {models.filter(m => m.inference_hz == null).length} models without reported speed
          </p>
        </div>
      </div>

      {/* Venue Distribution */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">
          Publication Venues
        </h4>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={venueDist} layout="vertical" margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
            <XAxis type="number" tick={{ fontSize: 10, fill: '#71717a' }} allowDecimals={false} />
            <YAxis dataKey="name" type="category" tick={{ fontSize: 10, fill: '#a1a1aa' }} width={80} />
            <Tooltip
              contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
            />
            <Bar dataKey="value" fill="#D4537E" radius={[0, 4, 4, 0]} barSize={14} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
