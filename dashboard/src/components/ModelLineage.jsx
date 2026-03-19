import { useState, useMemo } from 'react'

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
      { id: 'OpenVLA-OFT', label: 'OpenVLA-OFT', date: '2025-01', params: '7B', parent: 'OpenVLA', relation: 'orthogonal fine-tuning' },
      { id: 'OpenVLA-v2', label: 'OpenVLA-v2', date: '2025-02', params: '8B', parent: 'OpenVLA', relation: 'multi-modal action, SFT' },
    ],
  },
  {
    id: 'gr',
    name: 'ByteDance GR Series',
    color: '#D85A30',
    nodes: [
      { id: 'GR-1', label: 'GR-1', date: '2023-12', params: '130M' },
      { id: 'GR-2', label: 'GR-2', date: '2024-07', params: '1.5B', parent: 'GR-1', relation: 'video generation + action' },
      { id: 'DexVLA', label: 'DexVLA', date: '2025-02', params: '7B', parent: 'GR-2', relation: 'dexterous + hierarchical' },
      { id: 'HybridVLA', label: 'HybridVLA', date: '2025-03', params: '2B', parent: 'GR-2', relation: 'hybrid action space' },
    ],
  },
  {
    id: 'nvidia',
    name: 'NVIDIA',
    color: '#2ECC71',
    nodes: [
      { id: 'FLARE', label: 'FLARE', date: '2025-05', params: '3B' },
      { id: 'GR00T-N1', label: 'GR00T-N1', date: '2025-03', params: '2.2B' },
    ],
  },
  {
    id: 'google',
    name: 'Google DeepMind',
    color: '#3498DB',
    nodes: [
      { id: 'RT-2-X', label: 'RT-2-X', date: '2023-10', params: '55B' },
      { id: 'UniPi', label: 'UniPi', date: '2023-02', params: '—' },
    ],
  },
  {
    id: 'berkeley',
    name: 'UC Berkeley / Stanford',
    color: '#F39C12',
    nodes: [
      { id: 'Octo', label: 'Octo', date: '2024-05', params: '93M' },
      { id: 'CoT-VLA', label: 'CoT-VLA', date: '2025-01', params: '2B' },
      { id: 'SuSIE', label: 'SuSIE', date: '2023-10', params: '—' },
    ],
  },
  {
    id: 'shanghai',
    name: 'Shanghai AI Lab Ecosystem',
    color: '#D4537E',
    nodes: [
      { id: 'RoboVLM', label: 'RoboVLM', date: '2024-11', params: '1.5B' },
      { id: 'SpatialVLA', label: 'SpatialVLA', date: '2025-01', params: '4B', parent: 'RoboVLM', relation: 'spatial awareness' },
      { id: 'InstructVLA', label: 'InstructVLA', date: '2025-07', params: '4B', parent: 'SpatialVLA', relation: 'instruction-tuned' },
      { id: 'UD-VLA', label: 'UD-VLA', date: '2025-11', params: '3B' },
      { id: 'TRA-VLA', label: 'TRA-VLA', date: '2025-03', params: '4B' },
    ],
  },
]

function TreeNode({ node, family, depth, isLast, models }) {
  const modelData = models.find(m => m.name === node.id)
  const hasScore = modelData?.libero_avg != null

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
        <div
          className="flex items-center gap-2 px-3 py-2 rounded-lg border transition-all hover:scale-[1.02]"
          style={{
            borderColor: family.color + '40',
            backgroundColor: family.color + '08',
          }}
        >
          <div>
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
        </div>
      </div>
    </div>
  )
}

function FamilyTree({ family, models }) {
  // Build tree structure
  const roots = family.nodes.filter(n => !n.parent)
  const getChildren = (parentId) => family.nodes.filter(n => n.parent === parentId)

  function renderNode(node, depth = 0) {
    const children = getChildren(node.id)
    return (
      <div key={node.id}>
        <TreeNode node={node} family={family} depth={depth} models={models} />
        {children.length > 0 && (
          <div className="ml-6 pl-3 space-y-1.5 mt-1.5" style={{ borderLeft: `2px solid ${family.color}30` }}>
            {children.map(child => renderNode(child, depth + 1))}
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
    </div>
  )
}

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
      // Find max depth
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
          Evolution of VLA model families showing architectural lineage and improvements
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

      {/* Legend */}
      <div className="px-3 py-2 bg-zinc-900/50 border border-zinc-800/50 rounded-lg text-[10px] text-zinc-500">
        <strong className="text-zinc-400">Reading the tree:</strong> Each node shows model name, release date, parameter count, and LIBERO score (if available).
        Connecting lines show lineage relationships. Italic text describes the key architectural change.
      </div>
    </div>
  )
}
