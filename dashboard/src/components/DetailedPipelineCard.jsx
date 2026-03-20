import { Fragment } from 'react'
import { PIPELINE_CONFIGS } from '../data/pipelineConfigs'

// ── Color palette for dark-theme pipeline diagrams ──────────────────────────

const PAL = {
  b: { bg: 'rgba(59,130,246,0.10)', bd: 'rgba(59,130,246,0.35)', tx: '#93c5fd', st: 'rgba(147,197,253,0.50)' },
  p: { bg: 'rgba(139,92,246,0.10)', bd: 'rgba(139,92,246,0.35)', tx: '#c4b5fd', st: 'rgba(167,139,250,0.50)' },
  g: { bg: 'rgba(16,185,129,0.10)', bd: 'rgba(16,185,129,0.35)', tx: '#6ee7b7', st: 'rgba(52,211,153,0.50)' },
  o: { bg: 'rgba(249,115,22,0.10)', bd: 'rgba(249,115,22,0.35)', tx: '#fdba74', st: 'rgba(251,146,60,0.50)' },
  r: { bg: 'rgba(244,63,94,0.10)', bd: 'rgba(244,63,94,0.35)', tx: '#fda4af', st: 'rgba(251,113,133,0.50)' },
  t: { bg: 'rgba(20,184,166,0.10)', bd: 'rgba(20,184,166,0.35)', tx: '#5eead4', st: 'rgba(45,212,191,0.50)' },
  a: { bg: 'rgba(245,158,11,0.10)', bd: 'rgba(245,158,11,0.35)', tx: '#fcd34d', st: 'rgba(251,191,36,0.50)' },
  x: { bg: 'rgba(113,113,122,0.08)', bd: 'rgba(113,113,122,0.25)', tx: '#a1a1aa', st: 'rgba(113,113,122,0.50)' },
  i: { bg: 'rgba(99,102,241,0.10)', bd: 'rgba(99,102,241,0.35)', tx: '#a5b4fc', st: 'rgba(129,140,248,0.50)' },
  k: { bg: 'rgba(236,72,153,0.10)', bd: 'rgba(236,72,153,0.35)', tx: '#f9a8d4', st: 'rgba(244,114,182,0.50)' },
  e: { bg: 'rgba(5,150,105,0.14)', bd: 'rgba(5,150,105,0.45)', tx: '#6ee7b7', st: 'rgba(52,211,153,0.50)' },
  c: { bg: 'rgba(6,182,212,0.10)', bd: 'rgba(6,182,212,0.35)', tx: '#67e8f9', st: 'rgba(34,211,238,0.50)' },
  y: { bg: 'rgba(234,179,8,0.10)', bd: 'rgba(234,179,8,0.35)', tx: '#fde047', st: 'rgba(250,204,21,0.50)' },
}

function pc(color) { return PAL[color] || PAL.x }

// ── Sub-components ──────────────────────────────────────────────────────────

function DownArrow({ label }) {
  return (
    <div className="flex flex-col items-center py-0.5">
      <div className="w-px h-3 bg-zinc-700" />
      <svg width="8" height="5" className="text-zinc-600">
        <path d="M0 0 L4 4 L8 0" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
      {label && <div className="text-[7px] text-zinc-600 mt-0.5 text-center">{label}</div>}
    </div>
  )
}

function InputNode({ input }) {
  const c = pc(input.color)
  return (
    <div
      className="rounded-lg px-2 py-1 border text-center min-w-0"
      style={{ backgroundColor: c.bg, borderColor: c.bd }}
    >
      <div className="text-[9px] font-medium leading-tight" style={{ color: c.tx }}>{input.label}</div>
      {input.sub && <div className="text-[7px] leading-tight" style={{ color: c.st }}>{input.sub}</div>}
    </div>
  )
}

function SimpleStage({ stage }) {
  const c = pc(stage.color)
  return (
    <div
      className="rounded-lg px-3 py-1.5 border"
      style={{ backgroundColor: c.bg, borderColor: c.bd }}
    >
      <div className="text-[10px] font-semibold leading-tight" style={{ color: c.tx }}>{stage.label}</div>
      {stage.sub && <div className="text-[8px] leading-tight mt-0.5" style={{ color: c.st }}>{stage.sub}</div>}
    </div>
  )
}

function GroupStage({ stage }) {
  const c = pc(stage.color)
  return (
    <div
      className="rounded-lg border p-2"
      style={{ backgroundColor: c.bg, borderColor: c.bd }}
    >
      {/* Group title */}
      <div className="flex items-baseline gap-1.5 mb-1.5">
        <span className="text-[10px] font-semibold leading-tight" style={{ color: c.tx }}>{stage.group}</span>
        {stage.sub && <span className="text-[8px] leading-tight" style={{ color: c.st }}>{stage.sub}</span>}
      </div>

      {/* Children: rendered in rows if many, otherwise flex-wrap */}
      <div className="flex gap-1.5 flex-wrap justify-center">
        {stage.children.map((child, i) => {
          const cc = pc(child.color)
          return (
            <div
              key={i}
              className="rounded px-1.5 py-1 border text-center min-w-0"
              style={{ backgroundColor: cc.bg, borderColor: cc.bd }}
            >
              <div className="text-[8px] font-medium leading-tight" style={{ color: cc.tx }}>{child.label}</div>
              {child.sub && <div className="text-[7px] leading-tight" style={{ color: cc.st }}>{child.sub}</div>}
            </div>
          )
        })}
      </div>

      {/* Internal arrow + bottom child if present */}
      {stage.bottom && (
        <>
          <div className="flex justify-center py-0.5">
            <div className="w-px h-2 bg-zinc-700/50" />
          </div>
          {(() => {
            const bc = pc(stage.bottom.color)
            return (
              <div
                className="rounded px-2 py-1 border text-center"
                style={{ backgroundColor: bc.bg, borderColor: bc.bd }}
              >
                <div className="text-[9px] font-medium leading-tight" style={{ color: bc.tx }}>{stage.bottom.label}</div>
                {stage.bottom.sub && <div className="text-[7px] leading-tight" style={{ color: bc.st }}>{stage.bottom.sub}</div>}
              </div>
            )
          })()}
        </>
      )}
    </div>
  )
}

function OutputNode({ output }) {
  const c = pc(output.color)
  return (
    <div
      className="rounded-lg px-3 py-2 border text-center"
      style={{ backgroundColor: c.bg, borderColor: c.bd, boxShadow: `0 0 10px ${c.bd}` }}
    >
      <div className="text-[10px] font-semibold leading-tight" style={{ color: c.tx }}>{output.label}</div>
      {output.sub && <div className="text-[8px] leading-tight mt-0.5" style={{ color: c.st }}>{output.sub}</div>}
    </div>
  )
}

// ── Fallback config generator ───────────────────────────────────────────────

function generateFallbackConfig(model) {
  const arch = model.architecture || {}
  const inputs = [{ label: 'RGB frames', color: 'b' }]
  if (arch.llm && arch.llm !== 'null' && arch.llm !== 'None') {
    inputs.push({ label: 'Language', color: 'b' })
  }

  const stages = []
  if (arch.backbone) {
    stages.push({ label: arch.backbone, sub: 'vision encoder', color: 'p' })
  }
  if (arch.llm && arch.llm !== 'null' && arch.llm !== 'None') {
    stages.push({ label: arch.llm, sub: 'language model', color: 'i' })
  }
  stages.push({ label: arch.action_head || 'unknown', sub: 'action head', color: 'o' })

  return {
    inputs,
    stages,
    output: { label: 'Actions', color: 'e' },
  }
}

// ── Main component ──────────────────────────────────────────────────────────

export default function DetailedPipelineCard({ model }) {
  const config = PIPELINE_CONFIGS[model.name] || generateFallbackConfig(model)

  return (
    <div className="border border-zinc-800 rounded-xl overflow-hidden hover:border-zinc-600 transition-colors">
      {/* Header */}
      <div className="px-3 py-2 border-b border-zinc-800/50 flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="text-xs font-semibold text-white truncate">{model.name}</div>
          {model.organization && (
            <div className="text-[9px] text-zinc-600 truncate">{model.organization}</div>
          )}
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {model.inference_hz && (
            <span className="text-[9px] text-zinc-500 font-mono">{model.inference_hz}Hz</span>
          )}
          {model.architecture?.parameters && model.architecture.parameters !== 'unknown' && (
            <span className="text-[9px] text-zinc-500 font-mono bg-zinc-800/50 px-1.5 py-0.5 rounded">
              {model.architecture.parameters}
            </span>
          )}
        </div>
      </div>

      {/* Pipeline body */}
      <div className="px-3 py-3">
        {/* Inputs */}
        <div className="flex justify-center gap-1.5 flex-wrap">
          {config.inputs.map((inp, i) => <InputNode key={i} input={inp} />)}
        </div>

        <DownArrow />

        {/* Stages */}
        {config.stages.map((stage, i) => (
          <Fragment key={i}>
            {stage.group ? <GroupStage stage={stage} /> : <SimpleStage stage={stage} />}
            {i < config.stages.length - 1 ? (
              <DownArrow label={stage.arrowLabel} />
            ) : (
              <DownArrow />
            )}
          </Fragment>
        ))}

        {/* Output */}
        <OutputNode output={config.output} />
      </div>

      {/* Meta footer */}
      {config.meta && (
        <div className="px-3 py-1.5 border-t border-zinc-800/50 flex flex-wrap gap-x-3 gap-y-0.5">
          {config.meta.loss && (
            <div className="text-[9px]">
              <span className="text-zinc-600">Loss: </span>
              <span className="text-zinc-400">{config.meta.loss}</span>
            </div>
          )}
          {config.meta.loop && (
            <div className="text-[9px] text-amber-400/70">{config.meta.loop}</div>
          )}
          {config.meta.notes?.map((n, i) => (
            <div key={i} className="text-[9px] text-zinc-500">{n}</div>
          ))}
        </div>
      )}
    </div>
  )
}
