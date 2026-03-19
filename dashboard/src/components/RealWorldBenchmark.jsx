import { useMemo } from 'react'
import realWorldResults from '../data/realWorldResults.json'

const METRIC_COLORS = {
  success_rate: { bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
  throughput: { bg: 'bg-blue-500/10', text: 'text-blue-400' },
  failure_rate: { bg: 'bg-amber-500/10', text: 'text-amber-400' },
}

export default function RealWorldBenchmark({ models }) {
  const stats = useMemo(() => {
    const robots = [...new Set(realWorldResults.map(d => d.robot))].length
    const tasks = realWorldResults.reduce((s, d) => s + d.tasks.length, 0)
    return {
      models: realWorldResults.length,
      robots,
      tasks,
      latest: realWorldResults[0]?.date,
    }
  }, [])

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-sm font-semibold text-white">Real-World Benchmarks</h3>
        </div>
        <p className="text-[11px] text-zinc-500">
          Physical robot deployment results — tasks that simulation cannot capture.
          Unlike sim benchmarks, these reflect real-world noise, latency, and hardware constraints.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Models', value: stats.models, color: 'text-white' },
          { label: 'Robot Types', value: stats.robots, color: 'text-purple-400' },
          { label: 'Task Evals', value: stats.tasks, color: 'text-emerald-400' },
          { label: 'Latest', value: stats.latest, color: 'text-amber-400' },
        ].map(s => (
          <div key={s.label} className="border border-zinc-800 rounded-lg p-2 text-center">
            <div className={`text-base font-bold ${s.color}`}>{s.value}</div>
            <div className="text-[9px] text-zinc-500">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Model cards */}
      <div className="space-y-3">
        {realWorldResults.map(d => {
          const modelData = models.find(m => m.name === d.name)
          return (
            <div
              key={d.name}
              className={`border rounded-xl p-4 ${
                d.highlight
                  ? 'border-amber-500/30 bg-amber-500/5'
                  : 'border-zinc-800'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-white">{d.name}</span>
                    {d.highlight && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-medium">TOP</span>
                    )}
                    {modelData?.open_source && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-medium">OSS</span>
                    )}
                  </div>
                  <div className="text-[11px] text-zinc-500">{modelData?.organization || 'Unknown'} · {(modelData?.date || '').slice(0, 7)}</div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-zinc-500">{d.robot}</div>
                  <div className="text-[10px] text-zinc-600 italic">{d.method}</div>
                  <div className="flex items-center gap-1 justify-end mt-0.5">
                    {d.verified && (
                      <span className="text-[8px] px-1 py-0.5 rounded bg-blue-500/10 text-blue-400">paper-verified</span>
                    )}
                    <span className="text-[8px] text-zinc-600">{
                      modelData?.paper_url
                        ? `arXiv:${modelData.paper_url.match(/(\d{4}\.\d{4,5})/)?.[1] || d.source}`
                        : d.source
                    }</span>
                  </div>
                </div>
              </div>

              {/* Tasks */}
              <div className="space-y-1.5">
                {d.tasks.map((t, i) => {
                  const mc = METRIC_COLORS[t.metric] || METRIC_COLORS.success_rate
                  return (
                    <div key={i} className="flex items-center justify-between py-1 px-2 rounded-lg bg-zinc-900/50">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] text-zinc-300">{t.task}</span>
                        <span className="text-[9px] text-zinc-600">{t.setting}</span>
                      </div>
                      <span className={`text-[11px] font-semibold px-2 py-0.5 rounded ${mc.bg} ${mc.text}`}>
                        {t.value}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {/* Sim vs Real gap note */}
      <div className="px-3 py-2 bg-zinc-900/50 border border-zinc-800/50 rounded-lg text-[10px] text-zinc-500 leading-relaxed">
        <strong className="text-zinc-400">Note:</strong> Real-world results are not directly comparable across models
        due to different robots, tasks, and evaluation settings. Metrics include success rate, task throughput,
        and failure rate reduction. Results are extracted from published papers and may use different evaluation
        protocols. We are actively working on standardizing real-world evaluation reporting.
      </div>
    </div>
  )
}
