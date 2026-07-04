import { useMemo, useState, useRef, useCallback } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'

// ── Taxonomy ────────────────────────────────────────────────────────────────
// Decoding scheme (action_head_category) → cluster position + color.
// Fixed slot assignment (never cycled) — validated 8-slot dark palette.
const CATEGORIES = [
  { key: 'flow_matching',      label: 'Flow Matching',      color: '#3987e5' },
  { key: 'autoregressive',     label: 'Autoregressive',     color: '#199e70' },
  { key: 'diffusion',          label: 'Diffusion',          color: '#c98500' },
  { key: 'regression',         label: 'Regression',         color: '#008300' },
  { key: 'hybrid',             label: 'Hybrid (semi-AR)',   color: '#9085e9' },
  { key: 'discrete_diffusion', label: 'Discrete Diffusion', color: '#e66767' },
  { key: 'inverse_dynamics',   label: 'Inverse Dynamics',   color: '#d55181' },
  { key: 'other',              label: 'Other',              color: '#d95926' },
]
const CAT_INDEX = Object.fromEntries(CATEGORIES.map((c, i) => [c.key, i]))

// System paradigm (how the pieces connect) → node shape.
const PARADIGMS = [
  { key: 'e2e',     label: 'End-to-End',            desc: 'Backbone decodes actions directly',        shape: 'sphere' },
  { key: 'modular', label: 'VLM + Action Expert',   desc: 'Separate action module (DiT/expert head)', shape: 'cube' },
  { key: 'dual',    label: 'Dual-System',           desc: 'Fast/slow or hierarchical planner+actor',  shape: 'cone' },
  { key: 'wrapper', label: 'Wrapper / Test-time',   desc: 'Plug-in over a frozen base VLA',           shape: 'octa' },
]

const WRAPPER_RE = /wrapper|plug-?in|test-?time|steer|shield|recovery|routing|shared.?autonomy|corrector|frozen (base|vla|backbone)|post-?hoc|adapter genera|on top of/i
const DUAL_RE = /dual-?system|fast-?to-?slow|slow-?to-?fast|system ?[12]|hierarch|planner.{0,20}(controller|executor|actor)|high-?level.{0,25}low-?level/i
const MODULAR_RE = /action (expert|module|dit)|dit (action|expert|head)|flow.?matching (expert|head|dit|action)|expert (head|module)|separate (action|head)|diffusion (head|expert|transformer.{0,12}action)|\bdit\b/i

function deriveParadigm(m) {
  const arch = m.architecture || {}
  const text = [arch.action_head, arch.key_innovation, arch.backbone, (m.tags || []).join(' ')]
    .filter(Boolean).join(' | ')
  if (WRAPPER_RE.test(text)) return 'wrapper'
  if (DUAL_RE.test(text)) return 'dual'
  if (MODULAR_RE.test(text)) return 'modular'
  const cat = arch.action_head_category
  // Flow/diffusion families are, in practice, VLM + separate generative action module
  if (cat === 'flow_matching' || cat === 'diffusion' || cat === 'discrete_diffusion') return 'modular'
  return 'e2e'
}

// ── Layout ──────────────────────────────────────────────────────────────────
const CLUSTER_R = 17          // ring radius for the 8 clusters
const GOLDEN = Math.PI * (3 - Math.sqrt(5))
const Y_MIN = 40, Y_MAX = 100 // libero_avg → height 0..9

function liberoToY(avg) {
  if (avg == null) return 0.25
  const t = Math.min(1, Math.max(0, (avg - Y_MIN) / (Y_MAX - Y_MIN)))
  return 0.6 + t * 8.4
}

function buildNodes(models) {
  const byCat = new Map(CATEGORIES.map(c => [c.key, []]))
  for (const m of models) {
    const cat = m.architecture?.action_head_category || 'other'
    ;(byCat.get(cat) || byCat.get('other')).push(m)
  }
  const nodes = []
  for (const [cat, list] of byCat) {
    const ci = CAT_INDEX[cat]
    const angle = (ci / CATEGORIES.length) * Math.PI * 2
    const cx = Math.cos(angle) * CLUSTER_R
    const cz = Math.sin(angle) * CLUSTER_R
    list.sort((a, b) => (a.date || '').localeCompare(b.date || ''))
    list.forEach((m, i) => {
      const r = 0.5 * Math.sqrt(i + 0.6)
      const th = i * GOLDEN
      nodes.push({
        model: m,
        cat,
        paradigm: deriveParadigm(m),
        color: CATEGORIES[ci].color,
        x: cx + Math.cos(th) * r,
        z: cz + Math.sin(th) * r,
        y: liberoToY(m.libero_avg),
        hasScore: m.libero_avg != null,
      })
    })
  }
  return nodes
}

// ── 3D pieces ───────────────────────────────────────────────────────────────
function NodeMesh({ node, selected, dimmed, onHover, onClick }) {
  const ref = useRef()
  const scale = selected ? 1.6 : 1
  useFrame(() => {
    if (ref.current && selected) ref.current.rotation.y += 0.02
  })
  const common = {
    position: [node.x, node.y, node.z],
    scale,
    onPointerOver: (e) => { e.stopPropagation(); onHover(node) },
    onPointerOut: () => onHover(null),
    onClick: (e) => { e.stopPropagation(); onClick(node) },
  }
  const mat = (
    <meshStandardMaterial
      color={node.color}
      transparent
      opacity={dimmed ? 0.12 : node.hasScore ? 0.95 : 0.45}
      emissive={selected ? node.color : '#000000'}
      emissiveIntensity={selected ? 0.6 : 0}
      roughness={0.4}
    />
  )
  const s = node.hasScore ? 0.34 : 0.24
  switch (PARADIGMS.find(p => p.key === node.paradigm)?.shape) {
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

function ClusterLabel({ cat, count }) {
  const ci = CAT_INDEX[cat.key]
  const angle = (ci / CATEGORIES.length) * Math.PI * 2
  const x = Math.cos(angle) * (CLUSTER_R + 3.2)
  const z = Math.sin(angle) * (CLUSTER_R + 3.2)
  return (
    <Html position={[x, 0.2, z]} center zIndexRange={[10, 0]} style={{ pointerEvents: 'none' }}>
      <div className="pointer-events-none select-none text-center whitespace-nowrap">
        <div className="text-[11px] font-semibold" style={{ color: cat.color }}>{cat.label}</div>
        <div className="text-[10px] text-zinc-500">{count} models</div>
      </div>
    </Html>
  )
}

function Scene({ nodes, filters, hovered, setHovered, selected, onSelect }) {
  const counts = useMemo(() => {
    const c = {}
    nodes.forEach(n => { c[n.cat] = (c[n.cat] || 0) + 1 })
    return c
  }, [nodes])

  const isDimmed = useCallback((n) => {
    if (filters.cats.size && !filters.cats.has(n.cat)) return true
    if (filters.paradigms.size && !filters.paradigms.has(n.paradigm)) return true
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
          <NodeMesh
            node={n}
            selected={selected?.model.name === n.model.name}
            dimmed={isDimmed(n)}
            onHover={setHovered}
            onClick={onSelect}
          />
        </group>
      ))}
      {CATEGORIES.filter(c => counts[c.key]).map(c => (
        <ClusterLabel key={c.key} cat={c} count={counts[c.key]} />
      ))}
      {hovered && !isDimmed(hovered) && (
        <Html position={[hovered.x, hovered.y + 0.9, hovered.z]} center zIndexRange={[100, 0]} style={{ pointerEvents: 'none' }}>
          <div className="pointer-events-none whitespace-nowrap rounded-lg border border-zinc-700 bg-zinc-900/95 px-2.5 py-1.5 text-left shadow-xl">
            <div className="text-xs font-semibold text-white">{hovered.model.name}</div>
            <div className="text-[10px] text-zinc-400">
              {CATEGORIES[CAT_INDEX[hovered.cat]].label} · {PARADIGMS.find(p => p.key === hovered.paradigm)?.label}
            </div>
            <div className="text-[10px] text-zinc-300">
              {hovered.hasScore ? `LIBERO ${hovered.model.libero_avg.toFixed(1)}` : 'no LIBERO score'}
            </div>
          </div>
        </Html>
      )}
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={6}
        maxDistance={55}
        maxPolarAngle={Math.PI / 2.05}
        target={[0, 2.5, 0]}
      />
    </>
  )
}

// ── Shape glyphs for the legend (SVG, text stays in text tokens) ────────────
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
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] transition-all ${
        active
          ? 'border-zinc-500 bg-zinc-800 text-white'
          : 'border-zinc-800 bg-transparent text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {dotColor && <span className="h-2 w-2 rounded-full" style={{ background: dotColor }} />}
      {children}
    </button>
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

// ── Main component ──────────────────────────────────────────────────────────
export default function ArchitectureGalaxy({ models }) {
  const nodes = useMemo(() => buildNodes(models), [models])
  const [hovered, setHovered] = useState(null)
  const [selected, setSelected] = useState(null)
  const [compare, setCompare] = useState([])
  const [filters, setFilters] = useState({ cats: new Set(), paradigms: new Set(), scoredOnly: false })

  const toggle = (kind, key) => setFilters(f => {
    const next = new Set(f[kind])
    next.has(key) ? next.delete(key) : next.add(key)
    return { ...f, [kind]: next }
  })

  const addCompare = (node) => setCompare(prev =>
    prev.find(n => n.model.name === node.model.name)
      ? prev
      : [...prev.slice(-2), node]
  )

  const paradigmCounts = useMemo(() => {
    const c = {}
    nodes.forEach(n => { c[n.paradigm] = (c[n.paradigm] || 0) + 1 })
    return c
  }, [nodes])

  return (
    <div>
      {/* Filters — one row above the chart */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-zinc-500">Decoding</span>
        {CATEGORIES.map(c => (
          <Chip key={c.key} dotColor={c.color} active={!filters.cats.size || filters.cats.has(c.key)} onClick={() => toggle('cats', c.key)}>
            {c.label}
          </Chip>
        ))}
      </div>
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[11px] text-zinc-500">System</span>
        {PARADIGMS.map(p => (
          <Chip key={p.key} active={!filters.paradigms.size || filters.paradigms.has(p.key)} onClick={() => toggle('paradigms', p.key)}>
            <ShapeGlyph shape={p.shape} className="text-zinc-300" />
            {p.label} <span className="text-zinc-600">{paradigmCounts[p.key] || 0}</span>
          </Chip>
        ))}
        <Chip active={filters.scoredOnly} onClick={() => setFilters(f => ({ ...f, scoredOnly: !f.scoredOnly }))}>
          LIBERO scored only
        </Chip>
      </div>

      {/* Canvas + detail panel */}
      <div className="relative overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900">
        <Canvas camera={{ position: [0, 26, 38], fov: 45 }} style={{ height: 560 }} dpr={[1, 2]}>
          <Scene
            nodes={nodes}
            filters={filters}
            hovered={hovered}
            setHovered={setHovered}
            selected={selected}
            onSelect={setSelected}
          />
        </Canvas>

        {/* Height axis note */}
        <div className="pointer-events-none absolute left-3 top-3 text-[10px] leading-4 text-zinc-500">
          height = LIBERO avg ({Y_MIN}–{Y_MAX})<br />
          faded low nodes = no LIBERO score<br />
          drag to orbit · scroll to zoom
        </div>

        {/* Selected model panel */}
        {selected && (
          <div className="absolute right-3 top-3 w-72 max-h-[520px] overflow-y-auto rounded-xl border border-zinc-700 bg-zinc-900/95 p-3 shadow-2xl backdrop-blur">
            <div className="mb-1 flex items-start justify-between gap-2">
              <div className="text-sm font-bold text-white">{selected.model.name}</div>
              <button onClick={() => setSelected(null)} className="text-zinc-500 hover:text-white">✕</button>
            </div>
            <div className="mb-2 flex items-center gap-2 text-[11px]">
              <span className="flex items-center gap-1" style={{ color: selected.color }}>
                <span className="h-2 w-2 rounded-full" style={{ background: selected.color }} />
                {CATEGORIES[CAT_INDEX[selected.cat]].label}
              </span>
              <span className="flex items-center gap-1 text-zinc-300">
                <ShapeGlyph shape={PARADIGMS.find(p => p.key === selected.paradigm)?.shape} />
                {PARADIGMS.find(p => p.key === selected.paradigm)?.label}
              </span>
            </div>
            <DetailRow label="LIBERO avg" value={selected.model.libero_avg != null ? selected.model.libero_avg.toFixed(2) : '—'} />
            <DetailRow label="Backbone" value={selected.model.architecture?.backbone} />
            <DetailRow label="LLM" value={selected.model.architecture?.llm} />
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
                ['Decoding', n => CATEGORIES[CAT_INDEX[n.cat]].label],
                ['System', n => PARADIGMS.find(p => p.key === n.paradigm)?.label],
                ['LIBERO avg', n => n.model.libero_avg != null ? n.model.libero_avg.toFixed(2) : '—'],
                ['Backbone', n => n.model.architecture?.backbone || '—'],
                ['Action head', n => n.model.architecture?.action_head || '—'],
                ['Params', n => n.model.architecture?.parameters || '—'],
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
