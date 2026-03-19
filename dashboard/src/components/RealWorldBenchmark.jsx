import { useMemo } from 'react'

// Curated real-world deployment data from papers
const REAL_WORLD_DATA = [
  {
    name: 'pi*0.6',
    organization: 'Physical Intelligence',
    date: '2025-11',
    tasks: [
      { task: 'Laundry Folding', metric: 'throughput', value: '2x baseline', setting: 'Real homes' },
      { task: 'Box Assembly', metric: 'failure_rate', value: '50% reduction', setting: 'Warehouse' },
      { task: 'Espresso Making', metric: 'failure_rate', value: '50% reduction', setting: 'Professional machine' },
    ],
    method: 'RECAP (RL with corrections)',
    robot: 'Custom',
    highlight: true,
  },
  {
    name: 'pi0.5-Pro',
    organization: 'Physical Intelligence',
    date: '2025-09',
    tasks: [
      { task: 'Laundry Folding', metric: 'success_rate', value: '~80%', setting: 'Real homes' },
      { task: 'Table Bussing', metric: 'success_rate', value: '~85%', setting: 'Kitchen' },
      { task: 'Grocery Packing', metric: 'success_rate', value: '~75%', setting: 'Retail' },
    ],
    method: 'Cross-embodiment pre-training',
    robot: 'Multiple embodiments',
    highlight: true,
  },
  {
    name: 'pi0.5',
    organization: 'Physical Intelligence',
    date: '2025-02',
    tasks: [
      { task: 'Table Clearing', metric: 'success_rate', value: '~70%', setting: 'Kitchen' },
      { task: 'Object Rearrangement', metric: 'success_rate', value: '~65%', setting: 'Tabletop' },
    ],
    method: 'Cross-embodiment VLA',
    robot: 'Multiple embodiments',
  },
  {
    name: 'pi0',
    organization: 'Physical Intelligence',
    date: '2024-10',
    tasks: [
      { task: 'Laundry Folding', metric: 'success_rate', value: '~60%', setting: 'Lab' },
      { task: 'Table Bussing', metric: 'success_rate', value: '~55%', setting: 'Lab' },
    ],
    method: 'Flow matching VLA',
    robot: 'Multiple embodiments',
  },
  {
    name: 'GR00T-N1',
    organization: 'NVIDIA',
    date: '2025-03',
    tasks: [
      { task: 'Pick & Place', metric: 'success_rate', value: '~90%', setting: 'Lab (SO-100)' },
      { task: 'Reorientation', metric: 'success_rate', value: '~85%', setting: 'Lab (GR-1)' },
    ],
    method: 'Dual-system VLA',
    robot: 'SO-100 / GR-1',
  },
  {
    name: 'OpenVLA',
    organization: 'Stanford / UC Berkeley',
    date: '2024-06',
    tasks: [
      { task: 'WidowX Manipulation', metric: 'success_rate', value: '~55%', setting: 'Lab' },
      { task: 'RT-2 Benchmark Tasks', metric: 'success_rate', value: '~50%', setting: 'Lab (Google Robot)' },
    ],
    method: 'Fine-tuned VLM',
    robot: 'WidowX / Google Robot',
  },
  {
    name: 'RT-2-X',
    organization: 'Google DeepMind',
    date: '2023-10',
    tasks: [
      { task: 'Seen Objects Pick', metric: 'success_rate', value: '~73%', setting: 'Lab' },
      { task: 'Unseen Objects Pick', metric: 'success_rate', value: '~62%', setting: 'Lab' },
      { task: 'Drawer Manipulation', metric: 'success_rate', value: '~59%', setting: 'Lab' },
    ],
    method: 'VLM → action tokens',
    robot: 'Google Robot',
  },
  {
    name: 'Octo',
    organization: 'UC Berkeley / Stanford',
    date: '2024-05',
    tasks: [
      { task: 'WidowX Pick & Place', metric: 'success_rate', value: '~45%', setting: 'Lab' },
      { task: 'Franka Manipulation', metric: 'success_rate', value: '~40%', setting: 'Lab' },
    ],
    method: 'Generalist policy',
    robot: 'WidowX / Franka',
  },
]

const METRIC_COLORS = {
  success_rate: { bg: 'bg-emerald-500/10', text: 'text-emerald-400' },
  throughput: { bg: 'bg-blue-500/10', text: 'text-blue-400' },
  failure_rate: { bg: 'bg-amber-500/10', text: 'text-amber-400' },
}

export default function RealWorldBenchmark({ models }) {
  const stats = useMemo(() => {
    const robots = [...new Set(REAL_WORLD_DATA.map(d => d.robot))].length
    const tasks = REAL_WORLD_DATA.reduce((s, d) => s + d.tasks.length, 0)
    return {
      models: REAL_WORLD_DATA.length,
      robots,
      tasks,
      latest: REAL_WORLD_DATA[0]?.date,
    }
  }, [])

  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-sm font-semibold text-white">Real-World Benchmarks</h3>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 font-medium">NEW</span>
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
        {REAL_WORLD_DATA.map(d => {
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
                  <div className="text-[11px] text-zinc-500">{d.organization} · {d.date}</div>
                </div>
                <div className="text-right">
                  <div className="text-[10px] text-zinc-500">{d.robot}</div>
                  <div className="text-[10px] text-zinc-600 italic">{d.method}</div>
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
