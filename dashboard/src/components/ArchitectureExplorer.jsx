import { useMemo, useState } from 'react'
import { ACTION_HEAD_COLORS } from '../constants/benchmarks'
import DetailedPipelineCard from './DetailedPipelineCard'

// ── Normalizers ──────────────────────────────────────────────────────────────

function normalizeBackbone(raw) {
  if (!raw) return 'Unknown'
  const l = raw.toLowerCase()
  if (l.includes('prismatic') || (l.includes('siglip') && l.includes('dino'))) return 'PrismaticVLM'
  if (l.includes('paligemma') || (l.includes('siglip') && l.includes('gemma'))) return 'PaliGemma'
  if (l.includes('gemma') && !l.includes('siglip')) return 'PaliGemma'
  if (l.includes('internvl') || l.includes('internvit')) return 'InternVL'
  if (l.includes('qwen')) return 'Qwen-VL'
  if (l.includes('eagle') || l.includes('cosmos')) return 'Eagle VLM'
  if (l.includes('gemini')) return 'Gemini'
  if (l.includes('pali-x') || l.includes('palix') || l.includes('pali_x')) return 'PaLI-X'
  if (l.includes('florence')) return 'Florence'
  if (l.includes('kosmos') || l.includes('kosmo')) return 'Kosmos'
  if (l.includes('flamingo')) return 'OpenFlamingo'
  if (l.includes('siglip') && !l.includes('dino') && !l.includes('gemma')) return 'SigLIP'
  if (l.includes('clip') || l.includes('show-o')) return 'CLIP'
  if (l.includes('blip')) return 'BLIP'
  if (l.includes('mamba')) return 'Mamba'
  if (l.includes('diffusion') || l.includes('video') || l.includes('pix2pix')) return 'Video/Diffusion'
  if (l.includes('resnet') || l.includes('cnn')) return 'CNN'
  if (l.includes('sam2')) return 'SAM2'
  if (l.includes('transformer') || l.includes('gpt')) return 'Transformer'
  return 'Other'
}

function normalizeLLM(raw) {
  if (!raw) return 'None'
  const l = raw.toLowerCase()
  if (l.includes('llama-3') || l.includes('llama 3')) return 'Llama-3'
  if (l.includes('llama-2') || l.includes('llama 2') || l === 'llama') return 'Llama-2'
  if (l.includes('qwen')) return 'Qwen'
  if (l.includes('gemma')) return 'Gemma'
  if (l.includes('gpt')) return 'GPT'
  if (l.includes('gemini')) return 'Gemini'
  if (l.includes('eagle')) return 'Eagle'
  if (l.includes('t5') || l.includes('flan')) return 'T5/FlanT5'
  if (l.includes('phi')) return 'Phi'
  if (l.includes('mamba')) return 'Mamba'
  if (l.includes('internlm')) return 'InternLM'
  if (l.includes('smollm')) return 'SmolLM'
  if (l.includes('kosmos') || l.includes('kosmo')) return 'Kosmos'
  if (l === 'null' || l === 'none' || l === 'n/a') return 'None'
  return 'Other'
}

function normalizeActionHead(rawOrModel) {
  // If passed a model object, use action_head_category directly
  if (rawOrModel && typeof rawOrModel === 'object' && rawOrModel.architecture) {
    return rawOrModel.architecture.action_head_category || 'other'
  }
  // Legacy: string-based fallback
  if (!rawOrModel) return 'other'
  const l = String(rawOrModel).toLowerCase()
  const key = Object.keys(ACTION_HEAD_COLORS).find(k => l.includes(k))
  return key || 'other'
}

// ── Color Palettes ───────────────────────────────────────────────────────────

const BACKBONE_COLORS = {
  'PrismaticVLM': '#8b5cf6', 'PaliGemma': '#06b6d4', 'InternVL': '#f59e0b',
  'Qwen-VL': '#ef4444', 'Eagle VLM': '#22c55e', 'SigLIP': '#ec4899',
  'CLIP': '#3b82f6', 'Florence': '#a855f7', 'BLIP': '#14b8a6',
  'Gemini': '#4285f4', 'PaLI-X': '#ea4335', 'Kosmos': '#fbbf24', 'OpenFlamingo': '#f472b6',
  'Video/Diffusion': '#f97316', 'CNN': '#94a3b8', 'SAM2': '#fb923c',
  'Mamba': '#a78bfa', 'Transformer': '#38bdf8', 'Other': '#6b7280', 'Unknown': '#404040',
}

const LLM_COLORS = {
  'Llama-2': '#ef4444', 'Llama-3': '#f87171', 'Qwen': '#f59e0b',
  'Gemma': '#06b6d4', 'Gemini': '#3b82f6', 'Eagle': '#22c55e',
  'T5/FlanT5': '#a78bfa', 'GPT': '#10b981', 'Phi': '#ec4899',
  'Mamba': '#8b5cf6', 'InternLM': '#f97316', 'SmolLM': '#34d399',
  'Kosmos': '#fbbf24', 'Other': '#6b7280', 'None': '#404040',
}

const AH_COLORS = { ...ACTION_HEAD_COLORS, 'other': '#71717a' }

// ── Sankey-style Flow Diagram ────────────────────────────────────────────────

function FlowDiagram({ models, onSelectModels }) {
  const data = useMemo(() => {
    const backbones = {}, llms = {}, heads = {}
    const links = []

    models.forEach(m => {
      const bb = normalizeBackbone(m.architecture?.backbone)
      const llm = normalizeLLM(m.architecture?.llm)
      const ah = normalizeActionHead(m)

      backbones[bb] = (backbones[bb] || 0) + 1
      llms[llm] = (llms[llm] || 0) + 1
      heads[ah] = (heads[ah] || 0) + 1

      const k1 = `${bb}→${llm}`
      const k2 = `${llm}→${ah}`
      const existing1 = links.find(l => l.key === k1)
      if (existing1) { existing1.count++; existing1.models.push(m.name) }
      else links.push({ key: k1, from: bb, to: llm, col: 0, count: 1, models: [m.name] })

      const existing2 = links.find(l => l.key === k2)
      if (existing2) { existing2.count++; existing2.models.push(m.name) }
      else links.push({ key: k2, from: llm, to: ah, col: 1, count: 1, models: [m.name] })
    })

    const sortedBB = Object.entries(backbones).sort((a, b) => b[1] - a[1])
    const sortedLLM = Object.entries(llms).sort((a, b) => b[1] - a[1])
    const sortedAH = Object.entries(heads).sort((a, b) => b[1] - a[1])

    return { backbones: sortedBB, llms: sortedLLM, heads: sortedAH, links }
  }, [models])

  const [hovered, setHovered] = useState(null) // { col, name }

  const W = 700, H = Math.max(data.backbones.length, data.llms.length, data.heads.length) * 36 + 40
  const colX = [20, 280, 540]
  const nodeW = 130, nodeH = 26, nodeGap = 6

  function nodeY(items, idx) {
    const totalH = items.length * (nodeH + nodeGap) - nodeGap
    const startY = (H - totalH) / 2
    return startY + idx * (nodeH + nodeGap)
  }

  function getNodeCenter(col, name) {
    const items = col === 0 ? data.backbones : col === 1 ? data.llms : data.heads
    const idx = items.findIndex(([n]) => n === name)
    if (idx < 0) return { x: 0, y: 0 }
    const y = nodeY(items, idx)
    return { x: colX[col] + (col === 0 ? nodeW : col === 2 ? 0 : nodeW / 2), y: y + nodeH / 2 }
  }

  const isHighlighted = (col, name) => {
    if (!hovered) return true
    if (hovered.col === col && hovered.name === name) return true
    // highlight connected nodes
    return data.links.some(l => {
      if (hovered.col === 0) {
        if (l.col === 0 && l.from === hovered.name && (col === 1 && l.to === name)) return true
      }
      if (hovered.col === 1) {
        if (l.col === 0 && l.to === hovered.name && col === 0 && l.from === name) return true
        if (l.col === 1 && l.from === hovered.name && col === 2 && l.to === name) return true
      }
      if (hovered.col === 2) {
        if (l.col === 1 && l.to === hovered.name && (col === 1 && l.from === name)) return true
      }
      return false
    })
  }

  const isLinkHighlighted = (link) => {
    if (!hovered) return true
    if (link.col === 0) {
      if (hovered.col === 0 && link.from === hovered.name) return true
      if (hovered.col === 1 && link.to === hovered.name) return true
    }
    if (link.col === 1) {
      if (hovered.col === 1 && link.from === hovered.name) return true
      if (hovered.col === 2 && link.to === hovered.name) return true
    }
    return false
  }

  const maxLink = Math.max(...data.links.map(l => l.count), 1)

  const handleNodeClick = (col, name) => {
    const matching = models.filter(m => {
      if (col === 0) return normalizeBackbone(m.architecture?.backbone) === name
      if (col === 1) return normalizeLLM(m.architecture?.llm) === name
      return normalizeActionHead(m) === name
    })
    if (onSelectModels) onSelectModels(matching)
  }

  return (
    <div className="overflow-x-auto">
      {/* Column headers */}
      <div className="flex justify-between px-5 mb-2" style={{ maxWidth: W }}>
        <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider" style={{ width: nodeW, textAlign: 'center' }}>
          Vision Backbone
        </div>
        <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider" style={{ width: nodeW, textAlign: 'center' }}>
          Language Model
        </div>
        <div className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider" style={{ width: nodeW, textAlign: 'center' }}>
          Action Head
        </div>
      </div>

      <svg width={W} height={H} className="select-none" onMouseLeave={() => setHovered(null)}>
        {/* Links */}
        {data.links.map(link => {
          const fromCol = link.col
          const toCol = link.col + 1
          const from = getNodeCenter(fromCol, link.from)
          const to = getNodeCenter(toCol, link.to)
          const highlighted = isLinkHighlighted(link)
          const thickness = Math.max(1.5, (link.count / maxLink) * 8)

          return (
            <path
              key={link.key}
              d={`M${from.x},${from.y} C${from.x + 60},${from.y} ${to.x - 60},${to.y} ${to.x},${to.y}`}
              fill="none"
              stroke={fromCol === 0 ? (BACKBONE_COLORS[link.from] || '#555') : (LLM_COLORS[link.from] || '#555')}
              strokeWidth={thickness}
              opacity={highlighted ? 0.5 : 0.08}
              style={{ transition: 'opacity 0.2s' }}
            />
          )
        })}

        {/* Backbone nodes */}
        {data.backbones.map(([name, count], i) => {
          const y = nodeY(data.backbones, i)
          const hl = isHighlighted(0, name)
          return (
            <g
              key={name}
              onMouseEnter={() => setHovered({ col: 0, name })}
              onClick={() => handleNodeClick(0, name)}
              className="cursor-pointer"
            >
              <rect
                x={colX[0]} y={y} width={nodeW} height={nodeH} rx={6}
                fill={BACKBONE_COLORS[name] || '#555'}
                opacity={hl ? 0.85 : 0.15}
                style={{ transition: 'opacity 0.2s' }}
              />
              <text
                x={colX[0] + nodeW / 2} y={y + nodeH / 2}
                textAnchor="middle" dominantBaseline="central"
                fill="white" fontSize={10} fontWeight={500}
                opacity={hl ? 1 : 0.3}
                style={{ transition: 'opacity 0.2s', pointerEvents: 'none' }}
              >
                {name} ({count})
              </text>
            </g>
          )
        })}

        {/* LLM nodes */}
        {data.llms.map(([name, count], i) => {
          const y = nodeY(data.llms, i)
          const hl = isHighlighted(1, name)
          return (
            <g
              key={name}
              onMouseEnter={() => setHovered({ col: 1, name })}
              onClick={() => handleNodeClick(1, name)}
              className="cursor-pointer"
            >
              <rect
                x={colX[1]} y={y} width={nodeW} height={nodeH} rx={6}
                fill={LLM_COLORS[name] || '#555'}
                opacity={hl ? 0.85 : 0.15}
                style={{ transition: 'opacity 0.2s' }}
              />
              <text
                x={colX[1] + nodeW / 2} y={y + nodeH / 2}
                textAnchor="middle" dominantBaseline="central"
                fill="white" fontSize={10} fontWeight={500}
                opacity={hl ? 1 : 0.3}
                style={{ transition: 'opacity 0.2s', pointerEvents: 'none' }}
              >
                {name} ({count})
              </text>
            </g>
          )
        })}

        {/* Action Head nodes */}
        {data.heads.map(([name, count], i) => {
          const y = nodeY(data.heads, i)
          const hl = isHighlighted(2, name)
          return (
            <g
              key={name}
              onMouseEnter={() => setHovered({ col: 2, name })}
              onClick={() => handleNodeClick(2, name)}
              className="cursor-pointer"
            >
              <rect
                x={colX[2]} y={y} width={nodeW} height={nodeH} rx={6}
                fill={AH_COLORS[name] || '#555'}
                opacity={hl ? 0.85 : 0.15}
                style={{ transition: 'opacity 0.2s' }}
              />
              <text
                x={colX[2] + nodeW / 2} y={y + nodeH / 2}
                textAnchor="middle" dominantBaseline="central"
                fill="white" fontSize={10} fontWeight={500}
                opacity={hl ? 1 : 0.3}
                style={{ transition: 'opacity 0.2s', pointerEvents: 'none' }}
              >
                {name} ({count})
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// ── Architecture Cards ───────────────────────────────────────────────────────

function ArchitectureCard({ model }) {
  const bb = model.architecture?.backbone
  const llm = model.architecture?.llm
  const ah = model.architecture?.action_head
  const params = model.architecture?.parameters
  const bbNorm = normalizeBackbone(bb)
  const llmNorm = normalizeLLM(llm)
  const ahNorm = normalizeActionHead(ah)

  return (
    <div className="border border-zinc-800 rounded-xl p-3 hover:border-zinc-600 transition-colors">
      {/* Model name */}
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-white">{model.name}</span>
        {params && <span className="text-[10px] text-zinc-500 font-mono">{params}</span>}
      </div>

      {/* Pipeline visualization */}
      <div className="flex items-center gap-1.5">
        {/* Backbone */}
        <div
          className="flex-1 rounded-md px-2 py-1.5 text-center min-w-0"
          style={{ backgroundColor: `${BACKBONE_COLORS[bbNorm] || '#555'}25`, borderLeft: `3px solid ${BACKBONE_COLORS[bbNorm] || '#555'}` }}
        >
          <div className="text-[9px] text-zinc-500 leading-none mb-0.5">Vision</div>
          <div className="text-[10px] text-zinc-200 font-medium truncate" title={bb || 'Unknown'}>
            {bbNorm}
          </div>
        </div>

        <svg width="12" height="12" className="flex-shrink-0 text-zinc-600"><path d="M2 6h6M6 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5"/></svg>

        {/* LLM */}
        <div
          className="flex-1 rounded-md px-2 py-1.5 text-center min-w-0"
          style={{ backgroundColor: `${LLM_COLORS[llmNorm] || '#555'}25`, borderLeft: `3px solid ${LLM_COLORS[llmNorm] || '#555'}` }}
        >
          <div className="text-[9px] text-zinc-500 leading-none mb-0.5">LLM</div>
          <div className="text-[10px] text-zinc-200 font-medium truncate" title={llm || 'None'}>
            {llmNorm === 'None' ? '—' : llmNorm}
          </div>
        </div>

        <svg width="12" height="12" className="flex-shrink-0 text-zinc-600"><path d="M2 6h6M6 3l3 3-3 3" fill="none" stroke="currentColor" strokeWidth="1.5"/></svg>

        {/* Action Head */}
        <div
          className="flex-1 rounded-md px-2 py-1.5 text-center min-w-0"
          style={{ backgroundColor: `${AH_COLORS[ahNorm] || '#555'}25`, borderLeft: `3px solid ${AH_COLORS[ahNorm] || '#555'}` }}
        >
          <div className="text-[9px] text-zinc-500 leading-none mb-0.5">Action</div>
          <div className="text-[10px] text-zinc-200 font-medium truncate" title={ah || 'Unknown'}>
            {ahNorm}
          </div>
        </div>
      </div>

      {/* LIBERO score bar */}
      {model.libero_avg != null && (
        <div className="mt-2">
          <div className="flex justify-between text-[9px] text-zinc-500 mb-0.5">
            <span>LIBERO</span>
            <span>{model.libero_avg.toFixed(1)}%</span>
          </div>
          <div className="w-full h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full"
              style={{
                width: `${model.libero_avg}%`,
                backgroundColor: model.libero_avg >= 90 ? '#10b981' : model.libero_avg >= 70 ? '#f59e0b' : '#ef4444',
              }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

// ── Combination Matrix ───────────────────────────────────────────────────────

function CombinationMatrix({ models }) {
  const data = useMemo(() => {
    const combos = {}
    let maxCount = 0
    models.forEach(m => {
      const bb = normalizeBackbone(m.architecture?.backbone)
      const ah = normalizeActionHead(m)
      const key = `${bb}|${ah}`
      if (!combos[key]) combos[key] = { backbone: bb, actionHead: ah, count: 0, models: [], bestScore: null }
      combos[key].count++
      combos[key].models.push(m.name)
      if (m.libero_avg != null && (combos[key].bestScore == null || m.libero_avg > combos[key].bestScore)) {
        combos[key].bestScore = m.libero_avg
      }
      if (combos[key].count > maxCount) maxCount = combos[key].count
    })

    const backbones = [...new Set(Object.values(combos).map(c => c.backbone))].sort()
    const actionHeads = [...new Set(Object.values(combos).map(c => c.actionHead))].sort()

    return { combos, backbones, actionHeads, maxCount }
  }, [models])

  const [hoveredCell, setHoveredCell] = useState(null)

  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] w-full">
        <thead>
          <tr>
            <th className="text-left px-2 py-1.5 text-zinc-500 font-medium sticky left-0 bg-zinc-950 z-10">
              Backbone \ Action
            </th>
            {data.actionHeads.map(ah => (
              <th key={ah} className="text-center px-2 py-1.5 font-medium whitespace-nowrap">
                <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ backgroundColor: AH_COLORS[ah] || '#555' }} />
                <span className="text-zinc-400">{ah}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.backbones.map(bb => (
            <tr key={bb} className="border-t border-zinc-800/30">
              <td className="px-2 py-1.5 font-medium whitespace-nowrap sticky left-0 bg-zinc-950 z-10">
                <span className="inline-block w-2 h-2 rounded-full mr-1" style={{ backgroundColor: BACKBONE_COLORS[bb] || '#555' }} />
                <span className="text-zinc-300">{bb}</span>
              </td>
              {data.actionHeads.map(ah => {
                const key = `${bb}|${ah}`
                const cell = data.combos[key]
                const isHovered = hoveredCell === key
                return (
                  <td
                    key={ah}
                    className="text-center px-2 py-1.5 relative cursor-default"
                    onMouseEnter={() => setHoveredCell(key)}
                    onMouseLeave={() => setHoveredCell(null)}
                    style={{
                      backgroundColor: cell
                        ? `rgba(124, 58, 237, ${Math.min(0.15 + (cell.count / data.maxCount) * 0.55, 0.7)})`
                        : 'transparent',
                    }}
                  >
                    {cell ? (
                      <>
                        <span className="text-zinc-200 font-bold">{cell.count}</span>
                        {isHovered && (
                          <div className="absolute z-20 bottom-full left-1/2 -translate-x-1/2 mb-1 bg-zinc-900 border border-zinc-700 rounded-lg p-2 shadow-xl whitespace-nowrap">
                            <div className="text-[10px] text-zinc-300 font-semibold mb-1">{bb} + {ah}</div>
                            <div className="text-[9px] text-zinc-400">{cell.models.join(', ')}</div>
                            {cell.bestScore != null && (
                              <div className="text-[9px] text-emerald-400 mt-0.5">Best LIBERO: {cell.bestScore.toFixed(1)}%</div>
                            )}
                          </div>
                        )}
                      </>
                    ) : (
                      <span className="text-zinc-800">—</span>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function ArchitectureExplorer({ models }) {
  const [selectedModels, setSelectedModels] = useState(null)
  const [view, setView] = useState('flow') // 'flow' | 'matrix' | 'cards'

  // Stats
  const stats = useMemo(() => {
    const withBB = models.filter(m => m.architecture?.backbone).length
    const withLLM = models.filter(m => m.architecture?.llm).length
    const withAH = models.filter(m => m.architecture?.action_head).length
    const uniqueBB = new Set(models.map(m => normalizeBackbone(m.architecture?.backbone))).size
    const uniqueLLM = new Set(models.map(m => normalizeLLM(m.architecture?.llm))).size
    const uniqueAH = new Set(models.map(m => normalizeActionHead(m))).size
    return { withBB, withLLM, withAH, uniqueBB, uniqueLLM, uniqueAH }
  }, [models])

  const displayModels = selectedModels || models

  return (
    <div className="space-y-5">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-3">
        <div className="border border-zinc-800 rounded-xl p-3 text-center">
          <div className="text-lg font-bold text-purple-400">{stats.uniqueBB}</div>
          <div className="text-[10px] text-zinc-500">Backbone families</div>
          <div className="text-[9px] text-zinc-600">{stats.withBB}/{models.length} specified</div>
        </div>
        <div className="border border-zinc-800 rounded-xl p-3 text-center">
          <div className="text-lg font-bold text-cyan-400">{stats.uniqueLLM}</div>
          <div className="text-[10px] text-zinc-500">LLM families</div>
          <div className="text-[9px] text-zinc-600">{stats.withLLM}/{models.length} specified</div>
        </div>
        <div className="border border-zinc-800 rounded-xl p-3 text-center">
          <div className="text-lg font-bold text-amber-400">{stats.uniqueAH}</div>
          <div className="text-[10px] text-zinc-500">Action head types</div>
          <div className="text-[9px] text-zinc-600">{stats.withAH}/{models.length} specified</div>
        </div>
      </div>

      {/* View Toggle */}
      <div className="flex items-center gap-2">
        {[
          { id: 'flow', label: 'Flow Diagram' },
          { id: 'matrix', label: 'Combination Matrix' },
          { id: 'cards', label: 'Pipeline Cards' },
        ].map(v => (
          <button
            key={v.id}
            onClick={() => { setView(v.id); setSelectedModels(null) }}
            className={`px-3 py-1.5 text-xs rounded-lg border transition-all ${
              view === v.id
                ? 'border-zinc-600 bg-zinc-800 text-white font-semibold'
                : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`}
          >
            {v.label}
          </button>
        ))}
        {selectedModels && (
          <button
            onClick={() => setSelectedModels(null)}
            className="ml-auto px-3 py-1.5 text-xs rounded-lg border border-zinc-700 text-zinc-400 hover:text-white hover:border-zinc-500 transition-all"
          >
            Clear filter ({selectedModels.length} selected)
          </button>
        )}
      </div>

      {/* Flow View */}
      {view === 'flow' && (
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
            Architecture Flow
          </h4>
          <p className="text-[10px] text-zinc-600 mb-3">
            Hover to trace connections. Click a node to filter models. Line thickness = model count.
          </p>
          <FlowDiagram models={models} onSelectModels={setSelectedModels} />
        </div>
      )}

      {/* Matrix View */}
      {view === 'matrix' && (
        <div className="border border-zinc-800 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">
            Backbone × Action Head Matrix
          </h4>
          <p className="text-[10px] text-zinc-600 mb-3">
            Number of models using each combination. Hover for details.
          </p>
          <CombinationMatrix models={models} />
        </div>
      )}

      {/* Pipeline Cards View */}
      {view === 'cards' && (
        <>
          <p className="text-[10px] text-zinc-500">
            Showing {displayModels.length} models — each card shows the Vision → LLM → Action pipeline
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {displayModels
              .sort((a, b) => (b.libero_avg ?? -1) - (a.libero_avg ?? -1))
              .map(m => (
                <DetailedPipelineCard key={m.name} model={m} />
              ))}
          </div>
        </>
      )}
    </div>
  )
}
