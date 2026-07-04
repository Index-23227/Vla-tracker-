import { useMemo, useState, useRef, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'

// ── Validated 8-slot dark palette (fixed order, never cycled) ───────────────
const PALETTE = ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926']
const SHAPES = ['sphere', 'cube', 'cone', 'octa']

// ── Derivation helpers ──────────────────────────────────────────────────────
const archText = (m) => {
  const a = m.architecture || {}
  return [a.action_head, a.key_innovation, a.backbone, a.llm, (m.tags || []).join(' ')]
    .filter(Boolean).join(' | ')
}

const WRAPPER_RE = /wrapper|plug-?in|test-?time|steer|shield|recovery|routing|shared.?autonomy|corrector|frozen (base|vla|backbone)|post-?hoc|adapter genera|on top of/i
const DUAL_RE = /dual-?system|fast-?to-?slow|slow-?to-?fast|system ?[12]|hierarch|planner.{0,20}(controller|executor|actor)|high-?level.{0,25}low-?level/i
const MODULAR_RE = /action (expert|module|dit)|dit (action|expert|head)|flow.?matching (expert|head|dit|action)|expert (head|module)|separate (action|head)|diffusion (head|expert|transformer.{0,12}action)|\bdit\b/i

function deriveSystem(m) {
  const t = archText(m)
  if (WRAPPER_RE.test(t)) return 'Wrapper / Test-time'
  if (DUAL_RE.test(t)) return 'Dual-System'
  if (MODULAR_RE.test(t)) return 'VLM + Action Expert'
  const cat = m.architecture?.action_head_category
  if (cat === 'flow_matching' || cat === 'diffusion' || cat === 'discrete_diffusion') return 'VLM + Action Expert'
  return 'End-to-End'
}

const BACKBONE_RULES = [
  [/pi0[._-]?5|π0\.5|pi-?0\.5/i, 'π0.5'],
  [/pi0|π0|pi-?0\b/i, 'π0'],
  [/openvla/i, 'OpenVLA'],
  [/gr00t|groot/i, 'GR00T'],
  [/qwen/i, 'Qwen-VL'],
  [/paligemma/i, 'PaliGemma'],
  [/siglip|dinov?|vit\b|clip/i, 'ViT / custom'],
]
function deriveBackbone(m) {
  const a = m.architecture || {}
  const t = `${a.backbone || ''} ${a.llm || ''}`
  for (const [re, name] of BACKBONE_RULES) if (re.test(t)) return name
  return 'Other'
}

function deriveTraining(m) {
  const t = archText(m) + ' ' + ((m.training && JSON.stringify(m.training)) || '')
  if (/training-?free|test-?time|zero-?shot adapt/i.test(t)) return 'Test-time / Training-free'
  if (/grpo|\bppo\b|\brl\b|reinforcement|q-?learning|residual rl/i.test(t)) return 'RL post-training'
  if (/distill|quantiz|prun|compress|ptq|w4a4/i.test(t)) return 'Efficiency (distill/quant/prune)'
  return 'SFT / Imitation'
}

function parseParamsB(m) {
  const s = String(m.architecture?.parameters || '')
  const match = s.match(/(\d+(?:\.\d+)?)\s*([BM])/i)
  if (!match) return null
  const v = parseFloat(match[1])
  return match[2].toUpperCase() === 'B' ? v : v / 1000
}

function dateToMonths(m) {
  if (!m.date) return null
  const [y, mo] = m.date.split('-').map(Number)
  if (!y || !mo) return null
  return (y - 2022) * 12 + (mo - 1)
}

// ── Dimension registry ──────────────────────────────────────────────────────
// Categorical dims: drive cluster position/color OR node shape.
const CAT_DIMS = {
  decoding: {
    label: 'Decoding scheme',
    values: ['Flow Matching', 'Autoregressive', 'Diffusion', 'Regression', 'Hybrid (semi-AR)', 'Discrete Diffusion', 'Inverse Dynamics', 'Other'],
    of: (m) => ({
      flow_matching: 'Flow Matching', autoregressive: 'Autoregressive', diffusion: 'Diffusion',
      regression: 'Regression', hybrid: 'Hybrid (semi-AR)', discrete_diffusion: 'Discrete Diffusion',
      inverse_dynamics: 'Inverse Dynamics',
    }[m.architecture?.action_head_category] || 'Other'),
  },
  system: {
    label: 'System design',
    values: ['End-to-End', 'VLM + Action Expert', 'Dual-System', 'Wrapper / Test-time'],
    of: deriveSystem,
  },
  backbone: {
    label: 'Backbone family',
    values: ['π0.5', 'π0', 'OpenVLA', 'GR00T', 'Qwen-VL', 'PaliGemma', 'ViT / custom', 'Other'],
    of: deriveBackbone,
  },
  training: {
    label: 'Training paradigm',
    values: ['SFT / Imitation', 'RL post-training', 'Test-time / Training-free', 'Efficiency (distill/quant/prune)'],
    of: deriveTraining,
  },
  openSource: {
    label: 'Open source',
    values: ['Open source', 'Closed'],
    of: (m) => (m.open_source ? 'Open source' : 'Closed'),
  },
}
// Shape can encode at most 4 distinct values.
const SHAPE_DIM_KEYS = Object.keys(CAT_DIMS).filter(k => CAT_DIMS[k].values.length <= 4)

// Numeric dims: drive height.
const HEIGHT_DIMS = {
  libero: { label: 'LIBERO avg', of: m => m.libero_avg, domain: [40, 100], fmt: v => v.toFixed(1) },
  calvin: { label: 'CALVIN avg len', of: m => m.calvin_avg, domain: [0, 5], fmt: v => v.toFixed(2) },
  simpler: { label: 'SimplerEnv avg', of: m => m.simpler_avg, domain: [30, 90], fmt: v => v.toFixed(1) },
  robotwin_v2: { label: 'RoboTwin v2 avg', of: m => m.robotwin_v2_avg, domain: [0, 100], fmt: v => v.toFixed(1) },
  robocasa: { label: 'RoboCasa avg', of: m => m.robocasa_avg, domain: [0, 85], fmt: v => v.toFixed(1) },
  hz: { label: 'Inference Hz (log)', of: m => m.inference_hz, domain: [0, 3.1], log: true, fmt: v => `${v} Hz` },
  params: { label: 'Parameters (log B)', of: parseParamsB, domain: [-1.5, 1.3], log: true, fmt: v => `${v}B` },
  date: { label: 'Release date', of: dateToMonths, domain: [24, 55], fmt: (v, m) => m.date },
}

// ── Layout ──────────────────────────────────────────────────────────────────
const CLUSTER_R = 17
const GOLDEN = Math.PI * (3 - Math.sqrt(5))

function buildNodes(models, clusterKey, shapeKey, heightKey) {
  const cdim = CAT_DIMS[clusterKey]
  const sdim = CAT_DIMS[shapeKey]
  const hdim = HEIGHT_DIMS[heightKey]
  const byVal = new Map(cdim.values.map(v => [v, []]))
  for (const m of models) {
    const v = cdim.of(m)
    ;(byVal.get(v) || byVal.get(cdim.values[cdim.values.length - 1])).push(m)
  }
  const present = cdim.values.filter(v => byVal.get(v)?.length)
  const nodes = []
  present.forEach((val, pi) => {
    const list = byVal.get(val)
    const angle = (pi / present.length) * Math.PI * 2
    const cx = Math.cos(angle) * CLUSTER_R
    const cz = Math.sin(angle) * CLUSTER_R
    const color = PALETTE[cdim.values.indexOf(val) % 8]
    list.sort((a, b) => (a.date || '').localeCompare(b.date || ''))
    list.forEach((m, i) => {
      const r = 0.5 * Math.sqrt(i + 0.6)
      const th = i * GOLDEN
      let raw = hdim.of(m)
      if (raw != null && hdim.log) raw = Math.log10(Math.max(raw, 1e-6))
      let y = 0.25, hasScore = false
      if (raw != null) {
        hasScore = true
        const t = Math.min(1, Math.max(0, (raw - hdim.domain[0]) / (hdim.domain[1] - hdim.domain[0])))
        y = 0.6 + t * 8.4
      }
      nodes.push({
        model: m,
        clusterVal: val,
        shapeVal: sdim.of(m),
        shape: SHAPES[sdim.values.indexOf(sdim.of(m)) % 4],
        color, hasScore, y,
        heightRaw: hdim.of(m),
        x: cx + Math.cos(th) * r,
        z: cz + Math.sin(th) * r,
        clusterAngle: angle,
      })
    })
  })
  return { nodes, present }
}

// ── 3D pieces ───────────────────────────────────────────────────────────────
function NodeMesh({ node, selected, dimmed, onHover, onClick }) {
  const ref = useRef()
  useFrame(() => { if (ref.current && selected) ref.current.rotation.y += 0.02 })
  const common = {
    position: [node.x, node.y, node.z],
    scale: selected ? 1.6 : 1,
    onPointerOver: (e) => { e.stopPropagation(); onHover(node) },
    onPointerOut: () => onHover(null),
    onClick: (e) => { e.stopPropagation(); onClick(node) },
  }
  const mat = (
    <meshStandardMaterial
      color={node.color} transparent
      opacity={dimmed ? 0.12 : node.hasScore ? 0.95 : 0.45}
      emissive={selected ? node.color : '#000000'}
      emissiveIntensity={selected ? 0.6 : 0}
      roughness={0.4}
    />
  )
  const s = node.hasScore ? 0.34 : 0.24
  switch (node.shape) {
    case 'cube': return <mesh ref={ref} {...common}><boxGeometry args={[s * 1.5, s * 1.5, s * 1.5]} />{mat}</mesh>
    case 'cone': return <mesh ref={ref} {...common}><coneGeometry args={[s, s * 2.1, 12]} />{mat}</mesh>
    case 'octa': return <mesh ref={ref} {...common}><octahedronGeometry args={[s * 1.15]} />{mat}</mesh>
    default:     return <mesh ref={ref} {...common}><sphereGeometry args={[s, 20, 20]} />{mat}</mesh>
  }
}

function DropLine({ node, dimmed }) {
  if (!node.hasScore || node.y < 0.9) return null
  return (
    <mesh position={[node.x, node.y / 2, node.z]}>
      <cylinderGeometry args={[0.012, 0.012, node.y, 4]} />
      <meshBasicMaterial color={node.color} transparent opacity={dimmed ? 0.03 : 0.16} />
    </mesh>
  )
}

function Scene({ nodes, present, clusterKey, heightKey, filters, hovered, setHovered, selected, onSelect }) {
  const cdim = CAT_DIMS[clusterKey]
  const hdim = HEIGHT_DIMS[heightKey]
  const counts = useMemo(() => {
    const c = {}
    nodes.forEach(n => { c[n.clusterVal] = (c[n.clusterVal] || 0) + 1 })
    return c
  }, [nodes])

  const isDimmed = useCallback((n) => {
    if (filters.cluster.size && !filters.cluster.has(n.clusterVal)) return true
    if (filters.shape.size && !filters.shape.has(n.shapeVal)) return true
    if (filters.scoredOnly && !n.hasScore) return true
    return false
  }, [filters])

  return (
    <>
      <ambientLight intensity={0.55} />
      <directionalLight position={[10, 20, 8]} intensity={0.9} />
      <directionalLight position={[-12, 8, -10]} intensity={0.35} />
      <fog attach="fog" args={['#18181b', 38, 85]} />
      <gridHelper args={[52, 26, '#3f3f46', '#27272a']} position={[0, 0, 0]} />
      {nodes.map((n, i) => (
        <group key={n.model.name + i}>
          <DropLine node={n} dimmed={isDimmed(n)} />
          <NodeMesh node={n} selected={selected?.model.name === n.model.name}
                    dimmed={isDimmed(n)} onHover={setHovered} onClick={onSelect} />
        </group>
      ))}
      {present.map((val, pi) => {
        const angle = (pi / present.length) * Math.PI * 2
        const x = Math.cos(angle) * (CLUSTER_R + 3.2)
        const z = Math.sin(angle) * (CLUSTER_R + 3.2)
        const color = PALETTE[cdim.values.indexOf(val) % 8]
        return (
          <Html key={val} position={[x, 0.2, z]} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
            <div className="pointer-events-none select-none text-center whitespace-nowrap">
              <div className="text-[11px] font-semibold" style={{ color }}>{val}</div>
              <div className="text-[10px] text-zinc-500">{counts[val]} models</div>
            </div>
          </Html>
        )
      })}
      {hovered && !isDimmed(hovered) && (
        <Html position={[hovered.x, hovered.y + 0.9, hovered.z]} center zIndexRange={[100, 0]} style={{ pointerEvents: 'none' }}>
          <div className="pointer-events-none whitespace-nowrap rounded-lg border border-zinc-700 bg-zinc-900/95 px-2.5 py-1.5 text-left shadow-xl">
            <div className="text-xs font-semibold text-white">{hovered.model.name}</div>
            <div className="text-[10px] text-zinc-400">{hovered.clusterVal} · {hovered.shapeVal}</div>
            <div className="text-[10px] text-zinc-300">
              {hovered.heightRaw != null
                ? `${hdim.label.replace(/ \(.*\)/, '')}: ${hdim.fmt(hovered.heightRaw, hovered.model)}`
                : `no ${hdim.label.replace(/ \(.*\)/, '')} value`}
            </div>
          </div>
        </Html>
      )}
      <OrbitControls makeDefault enableDamping dampingFactor={0.08}
                     minDistance={6} maxDistance={55} maxPolarAngle={Math.PI / 2.05} target={[0, 2.5, 0]} />
    </>
  )
}

// ── DOM pieces ──────────────────────────────────────────────────────────────
function ShapeGlyph({ shape, className = '' }) {
  const c = 'currentColor'
  switch (shape) {
    case 'cube': return <svg viewBox="0 0 12 12" className={`h-3 w-3 ${className}`}><rect x="2" y="2" width="8" height="8" fill={c} /></svg>
    case 'cone': return <svg viewBox="0 0 12 12" className={`h-3 w-3 ${className}`}><polygon points="6,1 11,11 1,11" fill={c} /></svg>
    case 'octa': return <svg viewBox="0 0 12 12" className={`h-3 w-3 ${className}`}><polygon points="6,0.5 11.5,6 6,11.5 0.5,6" fill={c} /></svg>
    default:     return <svg viewBox="0 0 12 12" className={`h-3 w-3 ${className}`}><circle cx="6" cy="6" r="5" fill={c} /></svg>
  }
}

function Chip({ active, onClick, children, dotColor }) {
  return (
    <button onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-all ${
        active ? 'border-zinc-500 bg-zinc-800 text-white' : 'border-zinc-800 bg-transparent text-zinc-500 hover:text-zinc-300'
      }`}>
      {dotColor && <span className="h-2 w-2 rounded-full" style={{ background: dotColor }} />}
      {children}
    </button>
  )
}

function DimSelect({ label, value, onChange, options }) {
  return (
    <label className="flex items-center gap-1.5 text-[11px] text-zinc-500">
      {label}
      <select value={value} onChange={e => onChange(e.target.value)}
              className="rounded-md border border-zinc-700 bg-zinc-800 px-2 py-1 text-[11px] text-zinc-200 outline-none hover:border-zinc-600">
        {options.map(([k, l]) => <option key={k} value={k}>{l}</option>)}
      </select>
    </label>
  )
}

function DetailRow({ label, value }) {
  if (!value) return null
  return (
    <div className="mb-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="text-xs leading-relaxed text-zinc-200">{value}</div>
    </div>
  )
}

// ── Main ────────────────────────────────────────────────────────────────────
export default function ArchitectureGalaxy({ models }) {
  const [clusterKey, setClusterKey] = useState('decoding')
  const [shapeKey, setShapeKey] = useState('system')
  const [heightKey, setHeightKey] = useState('libero')
  const [hovered, setHovered] = useState(null)
  const [selected, setSelected] = useState(null)
  const [compare, setCompare] = useState([])
  const [filters, setFilters] = useState({ cluster: new Set(), shape: new Set(), scoredOnly: false })

  const { nodes, present } = useMemo(
    () => buildNodes(models, clusterKey, shapeKey, heightKey),
    [models, clusterKey, shapeKey, heightKey]
  )

  const cdim = CAT_DIMS[clusterKey]
  const sdim = CAT_DIMS[shapeKey]
  const hdim = HEIGHT_DIMS[heightKey]

  const changeDim = (setter) => (v) => {
    setter(v)
    setFilters({ cluster: new Set(), shape: new Set(), scoredOnly: false })
    setSelected(null); setHovered(null)
  }

  const toggle = (kind, key) => setFilters(f => {
    const next = new Set(f[kind])
    next.has(key) ? next.delete(key) : next.add(key)
    return { ...f, [kind]: next }
  })

  const addCompare = (node) => setCompare(prev =>
    prev.find(n => n.model.name === node.model.name) ? prev : [...prev.slice(-2), node])

  const shapeCounts = useMemo(() => {
    const c = {}
    nodes.forEach(n => { c[n.shapeVal] = (c[n.shapeVal] || 0) + 1 })
    return c
  }, [nodes])

  const scoredCount = nodes.filter(n => n.hasScore).length

  return (
    <div>
      {/* Dimension mapper */}
      <div className="mb-3 flex flex-wrap items-center gap-3 rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2">
        <DimSelect label="Cluster / color" value={clusterKey} onChange={changeDim(setClusterKey)}
                   options={Object.entries(CAT_DIMS).map(([k, d]) => [k, d.label])} />
        <DimSelect label="Shape" value={shapeKey} onChange={changeDim(setShapeKey)}
                   options={SHAPE_DIM_KEYS.map(k => [k, CAT_DIMS[k].label])} />
        <DimSelect label="Height" value={heightKey} onChange={changeDim(setHeightKey)}
                   options={Object.entries(HEIGHT_DIMS).map(([k, d]) => [k, d.label])} />
        <span className="text-[10px] text-zinc-600">{scoredCount}/{nodes.length} models have a {hdim.label.replace(/ \(.*\)/, '')} value</span>
      </div>

      {/* Filter chips for active dims */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-zinc-500">{cdim.label}</span>
        {present.map(v => (
          <Chip key={v} dotColor={PALETTE[cdim.values.indexOf(v) % 8]}
                active={!filters.cluster.size || filters.cluster.has(v)}
                onClick={() => toggle('cluster', v)}>
            {v}
          </Chip>
        ))}
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-zinc-500">{sdim.label}</span>
        {sdim.values.map((v, i) => (
          <Chip key={v} active={!filters.shape.size || filters.shape.has(v)} onClick={() => toggle('shape', v)}>
            <ShapeGlyph shape={SHAPES[i % 4]} className="text-zinc-300" />
            {v} <span className="text-zinc-600">{shapeCounts[v] || 0}</span>
          </Chip>
        ))}
        <Chip active={filters.scoredOnly} onClick={() => setFilters(f => ({ ...f, scoredOnly: !f.scoredOnly }))}>
          has {hdim.label.replace(/ \(.*\)/, '')}
        </Chip>
      </div>

      {/* Canvas */}
      <div className="relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900">
        <Canvas camera={{ position: [0, 26, 38], fov: 45 }} style={{ height: 560 }} dpr={[1, 2]}>
          <Scene nodes={nodes} present={present} clusterKey={clusterKey} heightKey={heightKey}
                 filters={filters} hovered={hovered} setHovered={setHovered}
                 selected={selected} onSelect={setSelected} />
        </Canvas>

        <div className="pointer-events-none absolute left-3 top-3 text-[10px] leading-4 text-zinc-500">
          height = {hdim.label}<br />
          faded low nodes = no value<br />
          drag to orbit · scroll to zoom
        </div>

        {selected && (
          <div className="absolute right-3 top-3 w-72 max-h-[520px] overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900/95 p-3 shadow-2xl backdrop-blur">
            <div className="mb-1 flex items-start justify-between gap-2">
              <div className="text-sm font-bold text-white">{selected.model.name}</div>
              <button onClick={() => setSelected(null)} className="text-zinc-500 hover:text-white">✕</button>
            </div>
            <div className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
              <span className="flex items-center gap-1" style={{ color: selected.color }}>
                <span className="h-2 w-2 rounded-full" style={{ background: selected.color }} />
                {selected.clusterVal}
              </span>
              <span className="flex items-center gap-1 text-zinc-300">
                <ShapeGlyph shape={selected.shape} />
                {selected.shapeVal}
              </span>
            </div>
            <DetailRow label={hdim.label} value={selected.heightRaw != null ? hdim.fmt(selected.heightRaw, selected.model) : '—'} />
            <DetailRow label="Decoding" value={CAT_DIMS.decoding.of(selected.model)} />
            <DetailRow label="System" value={CAT_DIMS.system.of(selected.model)} />
            <DetailRow label="Backbone family" value={CAT_DIMS.backbone.of(selected.model)} />
            <DetailRow label="Training" value={CAT_DIMS.training.of(selected.model)} />
            <DetailRow label="Backbone" value={selected.model.architecture?.backbone} />
            <DetailRow label="Action head" value={selected.model.architecture?.action_head} />
            <DetailRow label="Organization" value={selected.model.organization} />
            <DetailRow label="Date" value={selected.model.date} />
            <div className="mt-2 flex gap-2">
              {selected.model.paper_url && (
                <a href={selected.model.paper_url} target="_blank" rel="noopener noreferrer"
                   className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800">
                  Paper ↗
                </a>
              )}
              <button onClick={() => addCompare(selected)}
                      className="rounded-md border border-blue-500/40 bg-blue-500/10 px-2 py-1 text-[11px] text-blue-300 hover:bg-blue-500/20">
                + Compare
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Compare tray */}
      {compare.length > 0 && (
        <div className="mt-3 overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900 p-3">
          <div className="mb-2 flex items-center justify-between">
            <div className="text-xs font-semibold text-white">Architecture comparison</div>
            <button onClick={() => setCompare([])} className="text-[11px] text-zinc-500 hover:text-white">clear</button>
          </div>
          <table className="w-full min-w-[560px] text-left text-[11px]">
            <thead>
              <tr className="text-zinc-500">
                <th className="w-28 pb-1.5 pr-3 font-medium"> </th>
                {compare.map(n => (
                  <th key={n.model.name} className="pb-1.5 pr-3 font-semibold text-white">
                    <span className="mr-1.5 inline-block h-2 w-2 rounded-full align-middle" style={{ background: n.color }} />
                    {n.model.name}
                    <button onClick={() => setCompare(c => c.filter(x => x.model.name !== n.model.name))}
                            className="ml-1.5 text-zinc-600 hover:text-zinc-300">✕</button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="align-top text-zinc-300">
              {[
                ['Decoding', n => CAT_DIMS.decoding.of(n.model)],
                ['System', n => CAT_DIMS.system.of(n.model)],
                ['Backbone family', n => CAT_DIMS.backbone.of(n.model)],
                ['Training', n => CAT_DIMS.training.of(n.model)],
                ['LIBERO avg', n => n.model.libero_avg != null ? n.model.libero_avg.toFixed(2) : '—'],
                ['Backbone', n => n.model.architecture?.backbone || '—'],
                ['Action head', n => n.model.architecture?.action_head || '—'],
                ['Params', n => n.model.architecture?.parameters || '—'],
                ['Open source', n => n.model.open_source ? 'Yes' : 'No'],
                ['Date', n => n.model.date || '—'],
              ].map(([label, fn]) => (
                <tr key={label} className="border-t border-zinc-800">
                  <td className="py-1.5 pr-3 text-zinc-500">{label}</td>
                  {compare.map(n => <td key={n.model.name} className="py-1.5 pr-3">{fn(n)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
