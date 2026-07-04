import { PIPELINE_CONFIGS } from '../data/pipelineConfigs'

// ── Color palette for dark-theme pipeline diagrams ──────────────────────────

const PAL = {
  b: { bg: 'rgba(59,130,246,0.10)', bd: 'rgba(59,130,246,0.55)', tx: '#93c5fd', st: 'rgba(147,197,253,0.60)' },
  p: { bg: 'rgba(139,92,246,0.10)', bd: 'rgba(139,92,246,0.55)', tx: '#c4b5fd', st: 'rgba(167,139,250,0.60)' },
  g: { bg: 'rgba(16,185,129,0.10)', bd: 'rgba(16,185,129,0.55)', tx: '#6ee7b7', st: 'rgba(52,211,153,0.60)' },
  o: { bg: 'rgba(249,115,22,0.10)', bd: 'rgba(249,115,22,0.55)', tx: '#fdba74', st: 'rgba(251,146,60,0.60)' },
  r: { bg: 'rgba(244,63,94,0.10)', bd: 'rgba(244,63,94,0.55)', tx: '#fda4af', st: 'rgba(251,113,133,0.60)' },
  t: { bg: 'rgba(20,184,166,0.10)', bd: 'rgba(20,184,166,0.55)', tx: '#5eead4', st: 'rgba(45,212,191,0.60)' },
  a: { bg: 'rgba(245,158,11,0.10)', bd: 'rgba(245,158,11,0.55)', tx: '#fcd34d', st: 'rgba(251,191,36,0.60)' },
  x: { bg: 'rgba(113,113,122,0.08)', bd: 'rgba(113,113,122,0.45)', tx: '#a1a1aa', st: 'rgba(161,161,170,0.60)' },
  i: { bg: 'rgba(99,102,241,0.10)', bd: 'rgba(99,102,241,0.55)', tx: '#a5b4fc', st: 'rgba(129,140,248,0.60)' },
  k: { bg: 'rgba(236,72,153,0.10)', bd: 'rgba(236,72,153,0.55)', tx: '#f9a8d4', st: 'rgba(244,114,182,0.60)' },
  e: { bg: 'rgba(5,150,105,0.14)', bd: 'rgba(5,150,105,0.65)', tx: '#6ee7b7', st: 'rgba(52,211,153,0.60)' },
  c: { bg: 'rgba(6,182,212,0.10)', bd: 'rgba(6,182,212,0.55)', tx: '#67e8f9', st: 'rgba(34,211,238,0.60)' },
  y: { bg: 'rgba(234,179,8,0.10)', bd: 'rgba(234,179,8,0.55)', tx: '#fde047', st: 'rgba(250,204,21,0.60)' },
}
function pc(color) { return PAL[color] || PAL.x }

// ── Semantic role icons (inferred from stage text) ──────────────────────────
function roleIcon(stage) {
  const t = `${stage.label || ''} ${stage.group || ''} ${stage.sub || ''}`.toLowerCase()
  if (/memory|buffer|history|retriev/.test(t)) return 'stack'
  if (/world model|dynamics|predict|imagin|future|video gen/.test(t)) return 'globe'
  if (/plan|subgoal|keypoint|waypoint/.test(t)) return 'map'
  if (/diffusion|flow|denois|\bdit\b|action|decod|policy|head|expert|regress/.test(t)) return 'bolt'
  if (/llm|language|gemma|qwen|llama|gpt|vlm|paligemma|reason|cot|think/.test(t)) return 'chat'
  if (/vision|vit|siglip|dino|clip|encoder|image|point ?cloud|camera/.test(t)) return 'eye'
  return 'dot'
}

function RoleGlyph({ kind, color }) {
  const p = { stroke: color, strokeWidth: 1.4, fill: 'none', strokeLinecap: 'round', strokeLinejoin: 'round' }
  switch (kind) {
    case 'eye': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><path {...p} d="M1.5 8s2.5-4 6.5-4 6.5 4 6.5 4-2.5 4-6.5 4S1.5 8 1.5 8z" /><circle {...p} cx="8" cy="8" r="1.8" /></svg>
    case 'chat': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><path {...p} d="M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z" /></svg>
    case 'bolt': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><path {...p} d="M9 1.5 3.5 9H8l-1 5.5L12.5 7H8z" /></svg>
    case 'stack': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><ellipse {...p} cx="8" cy="4" rx="5.5" ry="2" /><path {...p} d="M2.5 4v8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2V4" /><path {...p} d="M2.5 8c0 1.1 2.5 2 5.5 2s5.5-.9 5.5-2" /></svg>
    case 'globe': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><circle {...p} cx="8" cy="8" r="6" /><path {...p} d="M2 8h12M8 2c-2 2-2 10 0 12M8 2c2 2 2 10 0 12" /></svg>
    case 'map': return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><path {...p} d="M3 13.5V4l3.5-1.5L10 4l3-1.5v9.5L10 13.5 6.5 12z" /><path {...p} d="M6.5 2.5V12M10 4v9.5" /></svg>
    default: return <svg viewBox="0 0 16 16" className="h-3.5 w-3.5"><circle cx="8" cy="8" r="3" fill={color} /></svg>
  }
}

// ── Flow pieces ─────────────────────────────────────────────────────────────

function InputPills({ inputs }) {
  return (
    <div className="flex flex-wrap justify-center gap-1.5">
      {inputs.map((inp, i) => {
        const c = pc(inp.color)
        return (
          <div key={i} className="rounded-full border px-2.5 py-1 text-center"
               style={{ background: c.bg, borderColor: c.bd }}>
            <span className="text-[10px] font-medium leading-none" style={{ color: c.tx }}>{inp.label}</span>
            {inp.sub && <span className="ml-1 text-[9px]" style={{ color: c.st }}>{inp.sub}</span>}
          </div>
        )
      })}
    </div>
  )
}

// A station on the spine: icon node on the left rail, card on the right.
function Station({ icon, color, children, terminal }) {
  return (
    <div className="relative flex gap-3 pl-1">
      <div className="relative z-10 mt-1 flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full border bg-zinc-900"
           style={{ borderColor: color, color, boxShadow: terminal ? `0 0 12px ${color}` : 'none' }}>
        <RoleGlyph kind={icon} color={color} />
      </div>
      <div className="min-w-0 flex-1 pb-1">{children}</div>
    </div>
  )
}

function ArrowLabel({ label }) {
  if (!label) return null
  return (
    <div className="relative z-10 my-0.5 flex items-center gap-2 pl-11">
      <span className="rounded-full border border-zinc-700/70 bg-zinc-900 px-2 py-0.5 text-[9px] text-zinc-400">
        {label}
      </span>
    </div>
  )
}

function StageCard({ stage }) {
  const c = pc(stage.color)
  const isGroup = !!stage.group
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/70 px-3 py-2"
         style={{ borderLeft: `3px solid ${c.bd}` }}>
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className="text-[12px] font-semibold leading-snug" style={{ color: c.tx }}>
          {stage.group || stage.label}
        </span>
        {stage.sub && <span className="text-[10px] leading-snug text-zinc-500">{stage.sub}</span>}
      </div>
      {isGroup && stage.children?.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {stage.children.map((child, i) => {
            const cc = pc(child.color)
            return (
              <div key={i} className="rounded-md border px-2 py-1"
                   style={{ background: cc.bg, borderColor: 'rgba(63,63,70,0.8)' }}>
                <div className="text-[10px] font-medium leading-tight" style={{ color: cc.tx }}>{child.label}</div>
                {child.sub && <div className="text-[9px] leading-tight text-zinc-500">{child.sub}</div>}
              </div>
            )
          })}
        </div>
      )}
      {stage.bottom && (() => {
        const bc = pc(stage.bottom.color)
        return (
          <div className="mt-1.5 rounded-md border border-dashed px-2 py-1"
               style={{ borderColor: bc.bd, background: bc.bg }}>
            <span className="text-[10px] font-medium" style={{ color: bc.tx }}>{stage.bottom.label}</span>
            {stage.bottom.sub && <span className="ml-1.5 text-[9px] text-zinc-500">{stage.bottom.sub}</span>}
          </div>
        )
      })()}
    </div>
  )
}

// ── Fallback config generator ───────────────────────────────────────────────

const clip = (s, n = 90) => (s && s.length > n ? s.slice(0, n - 1) + '…' : s)

function generateFallbackConfig(model) {
  const arch = model.architecture || {}
  const inputs = [{ label: 'RGB frames', color: 'b' }]
  const hasLLM = arch.llm && arch.llm !== 'null' && arch.llm !== 'None'
  if (hasLLM) inputs.push({ label: 'Language', color: 'b' })

  const stages = []
  if (arch.backbone) stages.push({ label: clip(arch.backbone), sub: 'vision encoder', color: 'p' })
  if (hasLLM) stages.push({ label: clip(arch.llm), sub: 'language model', color: 'i' })
  stages.push({ label: clip(arch.action_head) || 'unknown', sub: 'action head', color: 'o' })

  return { inputs, stages, output: { label: 'Actions', color: 'e' } }
}

// ── Main component ──────────────────────────────────────────────────────────

export default function DetailedPipelineCard({ model }) {
  const config = PIPELINE_CONFIGS[model.name] || generateFallbackConfig(model)
  const outC = pc(config.output.color)

  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800 transition-colors hover:border-zinc-600">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 border-b border-zinc-800/50 bg-zinc-900/60 px-3 py-2">
        <div className="min-w-0">
          <div className="truncate text-xs font-semibold text-white">{model.name}</div>
          {model.organization && (
            <div className="truncate text-[9px] text-zinc-600">{model.organization}</div>
          )}
        </div>
        <div className="flex flex-shrink-0 items-center gap-1.5">
          {model.inference_hz && (
            <span className="font-mono text-[9px] text-zinc-500">{model.inference_hz}Hz</span>
          )}
          {model.architecture?.parameters && model.architecture.parameters !== 'unknown' && (
            <span className="rounded bg-zinc-800/50 px-1.5 py-0.5 font-mono text-[9px] text-zinc-500">
              {clip(model.architecture.parameters, 24)}
            </span>
          )}
        </div>
      </div>

      {/* Pipeline body */}
      <div className="px-3 pb-3 pt-3">
        <InputPills inputs={config.inputs} />

        {/* Spine + stations */}
        <div className="relative mt-2">
          {/* vertical spine: gradient rail + animated pulse dot */}
          <div className="pointer-events-none absolute bottom-4 left-[17.5px] top-0 w-px"
               style={{ background: 'linear-gradient(to bottom, rgba(59,130,246,0.5), rgba(139,92,246,0.4), rgba(5,150,105,0.6))' }} />
          <div className="pipeline-pulse pointer-events-none absolute left-[15px] h-1.5 w-1.5 rounded-full"
               style={{ background: '#93c5fd', boxShadow: '0 0 6px #3b82f6' }} />

          {/* entry chevron from inputs */}
          <div className="relative z-10 mb-1 ml-[13px]">
            <svg width="10" height="6" className="text-zinc-600"><path d="M0 0 L5 5 L10 0" fill="none" stroke="currentColor" strokeWidth="1.5" /></svg>
          </div>

          <div className="flex flex-col gap-2">
            {config.stages.map((stage, i) => (
              <div key={i}>
                <Station icon={roleIcon(stage)} color={pc(stage.color).bd}>
                  <StageCard stage={stage} />
                </Station>
                <ArrowLabel label={stage.arrowLabel} />
              </div>
            ))}

            {/* Output terminal */}
            <Station icon="bolt" color={outC.bd} terminal>
              <div className="rounded-lg border px-3 py-2"
                   style={{ background: outC.bg, borderColor: outC.bd, boxShadow: `inset 0 0 20px ${outC.bg}` }}>
                <div className="text-[12px] font-semibold leading-snug" style={{ color: outC.tx }}>
                  {config.output.label}
                </div>
                {config.output.sub && (
                  <div className="mt-0.5 text-[10px] leading-snug" style={{ color: outC.st }}>{config.output.sub}</div>
                )}
              </div>
            </Station>
          </div>
        </div>
      </div>

      {/* Meta footer */}
      {config.meta && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 border-t border-zinc-800/50 px-3 py-1.5">
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
