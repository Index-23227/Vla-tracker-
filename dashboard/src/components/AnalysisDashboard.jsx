import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis, Cell,
  PieChart, Pie,
} from 'recharts'

const ACTION_HEAD_COLORS = {
  'flow matching': '#7F77DD',
  'autoregressive': '#1D9E75',
  'diffusion': '#D85A30',
  'diffusion transformer': '#D4537E',
  'chain-of-thought': '#F39C12',
  'unknown': '#71717a',
}

const PIE_COLORS = ['#7F77DD', '#1D9E75', '#D85A30', '#D4537E', '#3498DB', '#F39C12', '#E74C3C', '#71717a']

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
    const benches = [
      { key: 'libero_avg', label: 'LIBERO' },
      { key: 'calvin_avg', label: 'CALVIN' },
      { key: 'simpler_avg', label: 'SimplerEnv' },
      { key: 'robotwin_v1_avg', label: 'RoboTwin v1' },
      { key: 'robotwin_v2_avg', label: 'RoboTwin v2' },
      { key: 'rlbench_avg', label: 'RLBench' },
    ]
    return benches.map(b => ({
      name: b.label,
      count: models.filter(m => m[b.key] != null).length,
    }))
  }, [models])

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

      {/* Row 2: Parameter vs Performance */}
      <div className="border border-zinc-800 rounded-xl p-4">
        <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
          Parameters vs LIBERO Performance
        </h4>
        <p className="text-[10px] text-zinc-600 mb-3">Bubble size = parameter count · Green = open-source</p>
        {scatterData.length > 0 ? (
          <ResponsiveContainer width="100%" height={280}>
            <ScatterChart margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis
                dataKey="params"
                name="Parameters (M)"
                tick={{ fontSize: 10, fill: '#71717a' }}
                scale="log"
                domain={['auto', 'auto']}
                label={{ value: 'Parameters (M)', position: 'bottom', fontSize: 10, fill: '#52525b', offset: -5 }}
              />
              <YAxis
                dataKey="score"
                name="LIBERO Avg"
                tick={{ fontSize: 10, fill: '#71717a' }}
                domain={[40, 100]}
                label={{ value: 'LIBERO Avg (%)', angle: -90, position: 'insideLeft', fontSize: 10, fill: '#52525b' }}
              />
              <ZAxis dataKey="params" range={[40, 400]} />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #3f3f46', backgroundColor: '#18181b', color: '#e5e5e5' }}
                formatter={(value, name) => {
                  if (name === 'Parameters (M)') return [`${value}M`, name]
                  if (name === 'LIBERO Avg') return [`${value}%`, name]
                  return [value, name]
                }}
                labelFormatter={(_, payload) => payload?.[0]?.payload?.name || ''}
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
