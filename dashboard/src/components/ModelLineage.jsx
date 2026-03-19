import { useState, useMemo, useRef, useEffect } from 'react'

// Cross-family influence data
const CROSS_REFERENCES = [
  { from: 'RT-2-X', to: 'OpenVLA', label: 'RT-2 architecture inspired OpenVLA design' },
  { from: 'RT-2-X', to: 'pi0', label: 'RT-2 influence on early VLA paradigm' },
  { from: 'OpenVLA', to: 'SpatialVLA', label: 'OpenVLA framework adapted for spatial reasoning' },
  { from: 'Octo', to: 'OpenVLA', label: 'Octo data pipeline & cross-embodiment ideas' },
  { from: 'Octo', to: 'CrossFormer', label: 'CrossFormer built on Octo codebase' },
  { from: 'UniPi', to: 'SuSIE', label: 'Video prediction approach shared' },
  { from: 'GR-1', to: 'GR00T-N1', label: 'Humanoid-focused video-to-action paradigm' },
  { from: 'pi0', to: 'DexVLA', label: 'Flow matching action head approach' },
  { from: 'OpenVLA', to: 'ECoT', label: 'ECoT adds embodied CoT reasoning to OpenVLA' },
  { from: 'OpenVLA-OFT', to: 'VLA-Thinker', label: 'Visual CoT + GRPO RL on OpenVLA-OFT' },
]

// Model lineage data - curated family trees
const FAMILIES = [
  {
    id: 'pi',
    name: 'Physical Intelligence',
    color: '#7F77DD',
    nodes: [
      { id: 'pi0', label: 'pi0', date: '2024-10', params: '3B' },
      { id: 'pi0-FAST', label: 'pi0-FAST', date: '2025-01', params: '3B', parent: 'pi0', relation: 'FAST tokenizer variant' },
      { id: 'pi0.5', label: 'pi0.5', date: '2025-02', params: '3B+', parent: 'pi0', relation: 'cross-embodiment scaling' },
      { id: 'pi0.5-Pro', label: 'pi0.5-Pro', date: '2025-09', params: '5B+', parent: 'pi0.5', relation: 'larger model' },
      { id: 'pi*0.6', label: 'pi0.6', date: '2025-11', params: '3B+', parent: 'pi0', relation: 'RL (RECAP)' },
    ],
  },
  {
    id: 'openvla',
    name: 'OpenVLA (Stanford/Berkeley)',
    color: '#1D9E75',
    nodes: [
      { id: 'OpenVLA', label: 'OpenVLA', date: '2024-06', params: '7B' },
      { id: 'ECoT', label: 'ECoT', date: '2024-07', params: '7B', parent: 'OpenVLA', relation: 'embodied chain-of-thought' },
      { id: 'OpenVLA-OFT', label: 'OpenVLA-OFT', date: '2025-01', params: '7B', parent: 'OpenVLA', relation: 'orthogonal fine-tuning' },
      { id: 'OpenVLA-v2', label: 'OpenVLA-v2', date: '2025-02', params: '8B', parent: 'OpenVLA', relation: 'multi-modal action, SFT' },
      { id: 'VLA-Thinker', label: 'VLA-Thinker', date: '2026-03', params: '7B', parent: 'OpenVLA-OFT', relation: 'visual CoT + GRPO RL' },
    ],
  },
  {
    id: 'gr',
    name: 'ByteDance',
    color: '#D85A30',
    nodes: [
      { id: 'GR-1', label: 'GR-1', date: '2023-12', params: '130M' },
      { id: 'GR-2', label: 'GR-2', date: '2024-07', params: '1.5B', parent: 'GR-1', relation: 'video generation + action' },
      { id: 'DexVLA', label: 'DexVLA', date: '2025-02', params: '7B' },
      { id: 'HybridVLA', label: 'HybridVLA', date: '2025-03', params: '2B' },
    ],
  },
  {
    id: 'nvidia',
    name: 'NVIDIA',
    color: '#2ECC71',
    nodes: [
      { id: 'GR00T-N1', label: 'GR00T-N1', date: '2025-03', params: '2.2B' },
      { id: 'FLARE', label: 'FLARE', date: '2025-05', params: '3B' },
      { id: 'NaVILA', label: 'NaVILA', date: '2024-12', params: '8B' },
      { id: 'SAM2Act', label: 'SAM2Act', date: '2025-01', params: '—' },
    ],
  },
  {
    id: 'google',
    name: 'Google DeepMind',
    color: '#3498DB',
    nodes: [
      { id: 'UniPi', label: 'UniPi', date: '2023-02', params: '—' },
      { id: 'RT-2-X', label: 'RT-2-X', date: '2023-10', params: '55B' },
      { id: 'Mobility VLA', label: 'Mobility VLA', date: '2024-07', params: '—' },
      { id: 'Gemini Robotics', label: 'Gemini Robotics', date: '2025-03', params: '—' },
    ],
  },
  {
    id: 'berkeley',
    name: 'UC Berkeley / Stanford',
    color: '#F39C12',
    nodes: [
      { id: 'SuSIE', label: 'SuSIE', date: '2023-10', params: '—' },
      { id: 'Octo', label: 'Octo', date: '2024-05', params: '93M' },
      { id: 'LLARVA', label: 'LLARVA', date: '2024-06', params: '7B' },
      { id: 'CrossFormer', label: 'CrossFormer', date: '2024-08', params: '—', parent: 'Octo', relation: 'cross-embodiment scaling' },
      { id: 'CoT-VLA', label: 'CoT-VLA', date: '2025-01', params: '2B' },
    ],
  },
  {
    id: 'shanghai',
    name: 'Shanghai AI Lab Ecosystem',
    color: '#D4537E',
    nodes: [
      { id: 'DeeR-VLA', label: 'DeeR-VLA', date: '2024-11', params: '3B' },
      { id: 'RoboVLM', label: 'RoboVLM', date: '2024-11', params: '1.5B' },
      { id: 'SpatialVLA', label: 'SpatialVLA', date: '2025-01', params: '4B' },
      { id: 'TRA-VLA', label: 'TRA-VLA', date: '2025-03', params: '4B' },
      { id: 'InstructVLA', label: 'InstructVLA', date: '2025-07', params: '4B' },
      { id: 'RoboDual', label: 'RoboDual', date: '2024-10', params: '7B+20M' },
      { id: 'UD-VLA', label: 'UD-VLA', date: '2025-11', params: '3B' },
    ],
  },
]

// ─── Popover component for node click ───
function NodePopover({ node, modelData, family, onClose }) {
  const hasPaper = modelData?.paper_url
  const hasCode = modelData?.code_url
  const hasAny = hasPaper || hasCode

  return (
    <div
      className="absolute z-50 mt-1 min-w-[200px] bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl shadow-black/40 p-3"
      style={{ left: 0, top: '100%' }}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-bold text-white">{node.label}</span>
        <button onClick={onClose} className="text-zinc-500 hover:text-zinc-300 text-xs px-1">x</button>
      </div>
      <div className="space-y-1.5 text-[10px]">
        <div className="flex justify-between">
          <span className="text-zinc-500">Date</span>
          <span className="text-zinc-300">{node.date}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-zinc-500">Params</span>
          <span className="text-zinc-300">{node.params}</span>
        </div>
        {modelData?.organization && (
          <div className="flex justify-between gap-3">
            <span className="text-zinc-500 shrink-0">Org</span>
            <span className="text-zinc-300 text-right text-[9px]">{modelData.organization}</span>
          </div>
        )}
        {modelData?.architecture?.action_head && (
          <div className="flex justify-between gap-3">
            <span className="text-zinc-500 shrink-0">Action Head</span>
            <span className="text-zinc-300 text-right text-[9px]">{modelData.architecture.action_head}</span>
          </div>
        )}
        {modelData?.inference_hz && (
          <div className="flex justify-between">
            <span className="text-zinc-500">Speed</span>
            <span className="text-zinc-300">{modelData.inference_hz} Hz</span>
          </div>
        )}
        {modelData?.venue && (
          <div className="flex justify-between">
            <span className="text-zinc-500">Venue</span>
            <span className="text-zinc-300">{modelData.venue}</span>
          </div>
        )}
      </div>
      {hasAny && (
        <div className="flex gap-2 mt-2.5 pt-2 border-t border-zinc-800">
          {hasPaper && (
            <a
              href={modelData.paper_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 text-center text-[10px] font-medium px-2 py-1.5 rounded-md border transition-all"
              style={{
                borderColor: family.color + '50',
                color: family.color,
                backgroundColor: family.color + '10',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = family.color + '25' }}
              onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = family.color + '10' }}
            >
              Paper
            </a>
          )}
          {hasCode && (
            <a
              href={modelData.code_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 text-center text-[10px] font-medium px-2 py-1.5 rounded-md border border-emerald-500/40 text-emerald-400 bg-emerald-500/10 transition-all hover:bg-emerald-500/20"
            >
              Code
            </a>
          )}
        </div>
      )}
      {!hasAny && (
        <div className="mt-2 pt-2 border-t border-zinc-800 text-[9px] text-zinc-600 italic">
          No paper or code links available
        </div>
      )}
    </div>
  )
}

// ─── Tooltip component for hover ───
// CALVIN uses 0-5 scale, others use 0-100; normalize for display
const BENCHMARK_FIELDS = [
  { key: 'libero_avg', label: 'LIBERO', max: 100 },
  { key: 'calvin_avg', label: 'CALVIN', max: 5 },
  { key: 'robotwin_v1_avg', label: 'RoboTwin v1', max: 100 },
  { key: 'robotwin_v2_avg', label: 'RoboTwin v2', max: 100 },
  { key: 'simpler_avg', label: 'SimplerEnv', max: 100 },
]

function NodeTooltip({ node, modelData }) {
  const scores = BENCHMARK_FIELDS.filter(f => modelData?.[f.key] != null)

  return (
    <div className="absolute z-40 bottom-full left-1/2 -translate-x-1/2 mb-2 min-w-[180px] bg-zinc-900 border border-zinc-700 rounded-lg shadow-xl shadow-black/40 p-2.5 pointer-events-none">
      <div className="text-[10px] font-semibold text-white mb-1.5">{node.label}</div>
      {scores.length > 0 ? (
        <div className="space-y-1">
          {scores.map(s => {
            const raw = modelData[s.key]
            const pct = (raw / s.max) * 100
            return (
              <div key={s.key} className="flex items-center gap-1.5">
                <span className="text-[9px] text-zinc-500 w-16 shrink-0">{s.label}</span>
                <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${Math.min(pct, 100)}%`,
                      backgroundColor: pct >= 90 ? '#10b981' : pct >= 70 ? '#f59e0b' : '#ef4444',
                    }}
                  />
                </div>
                <span className="text-[9px] font-medium text-zinc-300 w-8 text-right">{raw.toFixed(1)}</span>
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-[9px] text-zinc-600 italic">No benchmark data</div>
      )}
      {modelData?.architecture?.action_head && (
        <div className="mt-1.5 pt-1.5 border-t border-zinc-800 text-[9px] text-zinc-500">
          <span className="text-zinc-400">Arch:</span> {modelData.architecture.action_head}
        </div>
      )}
      <div className="absolute left-1/2 -translate-x-1/2 top-full w-0 h-0 border-l-[5px] border-l-transparent border-r-[5px] border-r-transparent border-t-[5px] border-t-zinc-700" />
    </div>
  )
}

// ─── TreeNode with click popover, hover tooltip, collapse toggle ───
function TreeNode({ node, family, depth, models, childCount, isCollapsed, onToggleCollapse }) {
  const [showPopover, setShowPopover] = useState(false)
  const [showTooltip, setShowTooltip] = useState(false)
  const modelData = models.find(m => m.name === node.id)
  const hasScore = modelData?.libero_avg != null
  const hoverTimeout = useRef(null)
  const nodeRef = useRef(null)

  // Close popover on outside click (excluding the node itself to avoid toggle conflict)
  useEffect(() => {
    if (!showPopover) return
    function handleClickOutside(e) {
      if (nodeRef.current && nodeRef.current.contains(e.target)) return
      setShowPopover(false)
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [showPopover])

  // Cleanup hover timeout on unmount
  useEffect(() => {
    return () => clearTimeout(hoverTimeout.current)
  }, [])

  const handleMouseEnter = () => {
    hoverTimeout.current = setTimeout(() => setShowTooltip(true), 400)
  }
  const handleMouseLeave = () => {
    clearTimeout(hoverTimeout.current)
    setShowTooltip(false)
  }

  return (
    <div className="flex flex-col">
      <div className="flex items-stretch">
        {/* Connecting lines */}
        {depth > 0 && (
          <div className="flex items-center w-6 shrink-0">
            <div className="w-6 h-px" style={{ backgroundColor: family.color + '60' }} />
          </div>
        )}
        {/* Node */}
        <div className="relative" ref={nodeRef}>
          {showTooltip && !showPopover && modelData && (
            <NodeTooltip node={node} modelData={modelData} />
          )}
          <div
            className="flex items-center gap-2 px-3 py-2 rounded-lg border transition-all hover:scale-[1.02] cursor-pointer select-none"
            style={{
              borderColor: showPopover ? family.color + '80' : family.color + '40',
              backgroundColor: showPopover ? family.color + '18' : family.color + '08',
              boxShadow: showPopover ? `0 0 12px ${family.color}20` : 'none',
            }}
            onClick={() => { setShowPopover(!showPopover); setShowTooltip(false) }}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="text-xs font-semibold text-white">{node.label}</span>
                {modelData?.open_source && (
                  <span className="text-[8px] px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-medium">OSS</span>
                )}
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-[10px] text-zinc-500">{node.date}</span>
                <span className="text-[10px] text-zinc-600">·</span>
                <span className="text-[10px] text-zinc-500">{node.params}</span>
                {hasScore && (
                  <>
                    <span className="text-[10px] text-zinc-600">·</span>
                    <span className="text-[10px] font-medium text-emerald-400">
                      {modelData.libero_avg.toFixed(1)}
                    </span>
                  </>
                )}
              </div>
              {node.relation && (
                <div className="text-[9px] mt-0.5 italic" style={{ color: family.color + 'AA' }}>
                  {node.relation}
                </div>
              )}
            </div>
            {/* Collapse/expand toggle for nodes with children */}
            {childCount > 0 && (
              <button
                className="ml-1 w-5 h-5 flex items-center justify-center rounded text-[10px] transition-all hover:bg-zinc-700/50"
                style={{ color: family.color }}
                onClick={(e) => { e.stopPropagation(); onToggleCollapse() }}
                title={isCollapsed ? 'Expand children' : 'Collapse children'}
              >
                <svg
                  width="10" height="10" viewBox="0 0 10 10" fill="none"
                  className={`transition-transform duration-200 ${isCollapsed ? '-rotate-90' : ''}`}
                >
                  <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            )}
          </div>
          {showPopover && (
            <NodePopover
              node={node}
              modelData={modelData}
              family={family}
              onClose={() => setShowPopover(false)}
            />
          )}
        </div>
      </div>
    </div>
  )
}

// ─── Timeline bar for a family ───
function FamilyTimeline({ family }) {
  const dates = family.nodes.map(n => n.date).sort()
  const minDate = dates[0]
  const maxDate = dates[dates.length - 1]

  // Convert YYYY-MM to a numeric value for positioning
  const toNum = (d) => {
    const [y, m] = d.split('-').map(Number)
    return y * 12 + m
  }
  const minVal = toNum(minDate)
  const maxVal = toNum(maxDate)
  const range = maxVal - minVal || 1

  return (
    <div className="mt-3 pt-3 border-t border-zinc-800/50">
      <div className="text-[9px] text-zinc-600 mb-1.5">Timeline</div>
      <div className="relative h-6">
        {/* Track */}
        <div className="absolute top-1/2 -translate-y-1/2 left-0 right-0 h-px" style={{ backgroundColor: family.color + '30' }} />
        {/* Date labels at ends */}
        <div className="absolute -bottom-3 left-0 text-[8px] text-zinc-600">{minDate}</div>
        <div className="absolute -bottom-3 right-0 text-[8px] text-zinc-600">{maxDate}</div>
        {/* Nodes on timeline */}
        {family.nodes.map(node => {
          const pos = ((toNum(node.date) - minVal) / range) * 100
          const isRoot = !node.parent
          return (
            <div
              key={node.id}
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 group"
              style={{ left: `${Math.max(2, Math.min(98, pos))}%` }}
            >
              <div
                className="w-2.5 h-2.5 rounded-full border-2 transition-transform hover:scale-150"
                style={{
                  backgroundColor: isRoot ? family.color : 'transparent',
                  borderColor: family.color,
                }}
              />
              <div className="absolute bottom-full mb-1 left-1/2 -translate-x-1/2 hidden group-hover:block whitespace-nowrap bg-zinc-900 border border-zinc-700 rounded px-1.5 py-0.5 text-[8px] text-zinc-300 shadow-lg z-10">
                {node.label}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function sortByDate(nodes) {
  return [...nodes].sort((a, b) => a.date.localeCompare(b.date))
}

// ─── Family tree with collapse/expand support ───
function FamilyTree({ family, models }) {
  const [collapsed, setCollapsed] = useState({})

  const toggleCollapse = (nodeId) => {
    setCollapsed(prev => ({ ...prev, [nodeId]: !prev[nodeId] }))
  }

  // Build tree structure, sorted chronologically
  const roots = sortByDate(family.nodes.filter(n => !n.parent))
  const getChildren = (parentId) => sortByDate(family.nodes.filter(n => n.parent === parentId))

  // Count all descendants (not just direct children)
  const countDescendants = (nodeId) => {
    const children = family.nodes.filter(n => n.parent === nodeId)
    return children.reduce((sum, c) => sum + 1 + countDescendants(c.id), 0)
  }

  function renderNode(node, depth = 0) {
    const children = getChildren(node.id)
    const isCollapsed = collapsed[node.id] || false
    return (
      <div key={node.id}>
        <TreeNode
          node={node}
          family={family}
          depth={depth}
          models={models}
          childCount={children.length}
          isCollapsed={isCollapsed}
          onToggleCollapse={() => toggleCollapse(node.id)}
        />
        {children.length > 0 && !isCollapsed && (
          <div
            className="ml-6 pl-3 space-y-1.5 mt-1.5 transition-all"
            style={{ borderLeft: `2px solid ${family.color}30` }}
          >
            {children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
        {children.length > 0 && isCollapsed && (
          <div className="ml-6 pl-3 mt-1">
            <span
              className="text-[9px] px-1.5 py-0.5 rounded-full"
              style={{ color: family.color, backgroundColor: family.color + '15' }}
            >
              +{countDescendants(node.id)} collapsed
            </span>
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="border border-zinc-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-3">
        <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: family.color }} />
        <h4 className="text-xs font-semibold text-zinc-300">{family.name}</h4>
        <span className="text-[10px] text-zinc-600">({family.nodes.length} models)</span>
      </div>
      <div className="space-y-1.5">
        {roots.map(root => renderNode(root))}
      </div>
      {/* Timeline */}
      <FamilyTimeline family={family} />
    </div>
  )
}

// ─── Cross-family references section ───
function CrossReferences() {
  const [expanded, setExpanded] = useState(false)

  // Find family for a model id
  const findFamily = (modelId) => {
    for (const f of FAMILIES) {
      if (f.nodes.some(n => n.id === modelId)) return f
    }
    return null
  }

  return (
    <div className="border border-zinc-800 rounded-xl p-4">
      <button
        className="w-full flex items-center justify-between"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-blue-400">
            <path d="M2 7H5M9 7H12M7 2V5M7 9V12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="7" cy="7" r="1.5" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          <h4 className="text-xs font-semibold text-zinc-300">Cross-Family Influences</h4>
          <span className="text-[10px] text-zinc-600">({CROSS_REFERENCES.length} connections)</span>
        </div>
        <svg
          width="10" height="10" viewBox="0 0 10 10" fill="none"
          className={`text-zinc-500 transition-transform duration-200 ${expanded ? '' : '-rotate-90'}`}
        >
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {expanded && (
        <div className="mt-3 space-y-2">
          {CROSS_REFERENCES.map((ref, i) => {
            const fromFamily = findFamily(ref.from)
            const toFamily = findFamily(ref.to)
            return (
              <div key={i} className="flex items-center gap-2 text-[10px] px-2 py-1.5 rounded-lg bg-zinc-800/30">
                <span
                  className="px-1.5 py-0.5 rounded font-semibold"
                  style={{
                    color: fromFamily?.color || '#888',
                    backgroundColor: (fromFamily?.color || '#888') + '18',
                  }}
                >
                  {ref.from}
                </span>
                <svg width="16" height="8" viewBox="0 0 16 8" fill="none" className="text-zinc-600 shrink-0">
                  <path d="M0 4H14M14 4L10 1M14 4L10 7" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <span
                  className="px-1.5 py-0.5 rounded font-semibold"
                  style={{
                    color: toFamily?.color || '#888',
                    backgroundColor: (toFamily?.color || '#888') + '18',
                  }}
                >
                  {ref.to}
                </span>
                <span className="text-zinc-500 italic ml-1 truncate">{ref.label}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── Independent models section ───
function IndependentModels({ models }) {
  const [expanded, setExpanded] = useState(false)

  const familyModelIds = useMemo(() => {
    const ids = new Set()
    FAMILIES.forEach(f => f.nodes.forEach(n => ids.add(n.id)))
    return ids
  }, [])

  const independentModels = useMemo(() => {
    return models
      .filter(m => !familyModelIds.has(m.name))
      .sort((a, b) => (b.libero_avg || 0) - (a.libero_avg || 0))
  }, [models, familyModelIds])

  if (independentModels.length === 0) return null

  return (
    <div className="border border-zinc-800 rounded-xl p-4">
      <button
        className="w-full flex items-center justify-between"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" className="text-zinc-400">
            <rect x="2" y="2" width="4" height="4" rx="1" stroke="currentColor" strokeWidth="1.2" />
            <rect x="8" y="2" width="4" height="4" rx="1" stroke="currentColor" strokeWidth="1.2" />
            <rect x="5" y="8" width="4" height="4" rx="1" stroke="currentColor" strokeWidth="1.2" />
          </svg>
          <h4 className="text-xs font-semibold text-zinc-300">Independent Models</h4>
          <span className="text-[10px] text-zinc-600">({independentModels.length} models not in any family)</span>
        </div>
        <svg
          width="10" height="10" viewBox="0 0 10 10" fill="none"
          className={`text-zinc-500 transition-transform duration-200 ${expanded ? '' : '-rotate-90'}`}
        >
          <path d="M2 3.5L5 6.5L8 3.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {expanded && (
        <div className="mt-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1.5">
            {independentModels.map(m => (
              <div
                key={m.name}
                className="flex items-center justify-between px-2.5 py-1.5 rounded-lg border border-zinc-800/60 bg-zinc-800/20 hover:bg-zinc-800/40 transition-all"
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-[10px] font-semibold text-zinc-300 truncate">{m.name}</span>
                  {m.open_source && (
                    <span className="text-[7px] px-1 py-0.5 rounded bg-emerald-500/15 text-emerald-400 font-medium shrink-0">OSS</span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 shrink-0 ml-2">
                  <span className="text-[9px] text-zinc-600">{m.date?.slice(0, 7)}</span>
                  {m.libero_avg != null && (
                    <span className={`text-[9px] font-medium ${m.libero_avg >= 90 ? 'text-emerald-400' : m.libero_avg >= 70 ? 'text-amber-400' : 'text-zinc-500'}`}>
                      {m.libero_avg.toFixed(1)}
                    </span>
                  )}
                  {m.paper_url && (
                    <a
                      href={m.paper_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[8px] text-blue-400/60 hover:text-blue-400 transition-colors"
                      onClick={(e) => e.stopPropagation()}
                    >
                      paper
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Main export ───
export default function ModelLineage({ models }) {
  const [filter, setFilter] = useState('all')

  const filteredFamilies = useMemo(() => {
    if (filter === 'all') return FAMILIES
    return FAMILIES.filter(f => f.id === filter)
  }, [filter])

  // Stats
  const stats = useMemo(() => {
    const totalInFamilies = FAMILIES.reduce((s, f) => s + f.nodes.length, 0)
    const orphans = models.length - totalInFamilies
    const familyCount = FAMILIES.length
    const longestChain = FAMILIES.reduce((max, f) => {
      const getDepth = (nodeId, d = 0) => {
        const children = f.nodes.filter(n => n.parent === nodeId)
        if (children.length === 0) return d
        return Math.max(...children.map(c => getDepth(c.id, d + 1)))
      }
      const roots = f.nodes.filter(n => !n.parent)
      const maxDepth = roots.length > 0 ? Math.max(...roots.map(r => getDepth(r.id, 0))) : 0
      return Math.max(max, maxDepth)
    }, 0)
    return { totalInFamilies, orphans, familyCount, longestChain }
  }, [models])

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-semibold text-white mb-1">Model Lineage Graph</h3>
        <p className="text-[11px] text-zinc-500">
          Evolution of VLA model families showing architectural lineage and improvements. Click nodes for details.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: 'Families', value: stats.familyCount, color: 'text-purple-400' },
          { label: 'In Lineage', value: stats.totalInFamilies, color: 'text-emerald-400' },
          { label: 'Independent', value: stats.orphans, color: 'text-zinc-400' },
          { label: 'Max Depth', value: stats.longestChain, color: 'text-amber-400' },
        ].map(s => (
          <div key={s.label} className="border border-zinc-800 rounded-lg p-2 text-center">
            <div className={`text-base font-bold ${s.color}`}>{s.value}</div>
            <div className="text-[9px] text-zinc-500">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Family filter */}
      <div className="flex gap-1.5 flex-wrap">
        <button
          onClick={() => setFilter('all')}
          className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
            filter === 'all'
              ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
              : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
          }`}
        >
          All Families
        </button>
        {FAMILIES.map(f => (
          <button
            key={f.id}
            onClick={() => setFilter(f.id)}
            className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
              filter === f.id
                ? 'border-zinc-500 bg-zinc-800 text-white font-semibold'
                : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
            }`}
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full mr-1" style={{ backgroundColor: f.color }} />
            {f.name}
          </button>
        ))}
      </div>

      {/* Family trees */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {filteredFamilies.map(family => (
          <FamilyTree key={family.id} family={family} models={models} />
        ))}
      </div>

      {/* Cross-family influences */}
      <CrossReferences />

      {/* Independent models */}
      <IndependentModels models={models} />

      {/* Legend */}
      <div className="px-3 py-2 bg-zinc-900/50 border border-zinc-800/50 rounded-lg text-[10px] text-zinc-500">
        <strong className="text-zinc-400">Reading the tree:</strong> Click any node to see paper/code links and details.
        Hover to preview benchmark scores. Use the chevron (v) to collapse/expand branches.
        Timeline bar at the bottom of each family shows temporal progression.
        Filled circles = root models, hollow circles = derived models.
      </div>
    </div>
  )
}
