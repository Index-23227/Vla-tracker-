import { useState, useMemo } from 'react'

function SimpleMarkdownCompact({ text }) {
  if (!text) return null
  const lines = text.split('\n')
  const elements = []
  let inTable = false
  let tableRows = []

  const flushTable = () => {
    if (tableRows.length === 0) return
    const headerCells = tableRows[0]
    const bodyRows = tableRows.slice(2)
    elements.push(
      <div key={`t${elements.length}`} className="overflow-x-auto my-2">
        <table className="w-full text-xs border-collapse">
          <thead>
            <tr>{headerCells.map((c, i) => <th key={i} className="text-left px-2 py-1 border-b border-zinc-700 text-zinc-400 font-medium">{c.trim()}</th>)}</tr>
          </thead>
          <tbody>
            {bodyRows.map((row, ri) => (
              <tr key={ri} className="border-b border-zinc-800/50">
                {row.map((c, ci) => <td key={ci} className="px-2 py-1 text-zinc-300">{c.trim()}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
    tableRows = []
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (line.startsWith('```') || line.startsWith('<!--')) continue

    if (line.startsWith('|') && line.includes('|')) {
      if (!inTable) inTable = true
      if (line.match(/^\|[\s-:|]+\|$/)) { tableRows.push(null); continue }
      tableRows.push(line.split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1))
      continue
    } else if (inTable) { inTable = false; flushTable() }

    if (line.startsWith('# ')) { elements.push(<h1 key={i} className="text-base font-bold text-white mt-4 mb-2">{line.slice(2)}</h1>); continue }
    if (line.startsWith('## ')) { elements.push(<h2 key={i} className="text-sm font-bold text-white mt-3 mb-1 border-b border-zinc-800 pb-1">{line.slice(3)}</h2>); continue }
    if (line.startsWith('### ')) { elements.push(<h3 key={i} className="text-xs font-bold text-zinc-300 mt-2 mb-1">{line.slice(4)}</h3>); continue }
    if (line.startsWith('> ')) {
      const isQ = line.includes('❓') || line.includes('⚠️')
      elements.push(<blockquote key={i} className={`border-l-2 pl-3 my-1 text-xs ${isQ ? 'border-amber-500/50 text-amber-200/80' : 'border-purple-500/50 text-zinc-400 italic'}`}>{line.slice(2)}</blockquote>)
      continue
    }
    if (line.startsWith('- ')) { elements.push(<li key={i} className="text-xs text-zinc-300 ml-4 list-disc mb-0.5">{line.slice(2)}</li>); continue }
    if (line.startsWith('---')) { elements.push(<hr key={i} className="border-zinc-800 my-2" />); continue }
    if (line.trim() === '') continue

    elements.push(<p key={i} className="text-xs text-zinc-300 mb-1">{line.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').split('<b>').map((part, pi) => {
      if (pi === 0) return part
      const [bold, rest] = part.split('</b>')
      return <span key={pi}><strong className="text-white">{bold}</strong>{rest}</span>
    })}</p>)
  }
  if (inTable) flushTable()
  return <div>{elements}</div>
}

function extractSummary(content) {
  const match = content?.match(/^>.*?한 줄 요약.*?:\s*(.+)$/m)
  return match ? match[1].trim() : null
}

function extractRating(content) {
  const novelty = content?.match(/Novelty.*?([★☆]{5})/)?.[1]
  const impact = content?.match(/Practical impact.*?([★☆]{5})/)?.[1]
  return { novelty, impact }
}

export default function PaperReviews({ models }) {
  const [search, setSearch] = useState('')
  const [filter, setFilter] = useState('all') // all, verified, abstract
  const [expandedModel, setExpandedModel] = useState(null)

  const reviewModels = useMemo(() => {
    return models
      .filter(m => m.ai_review?.content)
      .map(m => ({
        ...m,
        summary: extractSummary(m.ai_review.content),
        ratings: extractRating(m.ai_review.content),
      }))
  }, [models])

  const filtered = useMemo(() => {
    let result = reviewModels
    if (filter === 'verified') result = result.filter(m => m.ai_review.verified)
    if (filter === 'abstract') result = result.filter(m => !m.ai_review.verified)
    if (search) {
      const q = search.toLowerCase()
      result = result.filter(m =>
        m.name.toLowerCase().includes(q) ||
        m.organization?.toLowerCase().includes(q) ||
        m.ai_review.content.toLowerCase().includes(q)
      )
    }
    return result
  }, [reviewModels, filter, search])

  const verifiedCount = reviewModels.filter(m => m.ai_review.verified).length

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg font-bold text-white">Paper Reviews</h2>
          <p className="text-xs text-zinc-500 mt-0.5">
            {reviewModels.length} reviews ({verifiedCount} PDF-verified)
          </p>
        </div>
      </div>

      {/* Search & Filter */}
      <div className="flex gap-2 mb-4">
        <input
          type="text"
          placeholder="Search papers, methods, keywords..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="flex-1 px-3 py-2 text-sm bg-zinc-800 border border-zinc-700 rounded-lg text-white placeholder-zinc-500 focus:outline-none focus:border-zinc-500"
        />
        <div className="flex gap-1">
          {[
            { id: 'all', label: 'All' },
            { id: 'verified', label: 'PDF Verified' },
            { id: 'abstract', label: 'Abstract' },
          ].map(f => (
            <button
              key={f.id}
              onClick={() => setFilter(f.id)}
              className={`px-3 py-2 text-xs rounded-lg border transition-colors ${
                filter === f.id
                  ? 'border-zinc-600 bg-zinc-700 text-white'
                  : 'border-zinc-800 text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Results count */}
      <div className="text-xs text-zinc-600 mb-3">{filtered.length} papers</div>

      {/* Review Cards */}
      <div className="space-y-2">
        {filtered.map(model => {
          const isExpanded = expandedModel === model.name

          return (
            <div key={model.name} className="border border-zinc-800 rounded-xl overflow-hidden">
              {/* Card Header (always visible) */}
              <button
                onClick={() => setExpandedModel(isExpanded ? null : model.name)}
                className="w-full px-4 py-3 flex items-start justify-between text-left hover:bg-zinc-800/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-semibold text-white">{model.name}</span>
                    {model.ai_review.verified ? (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 shrink-0">PDF</span>
                    ) : (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-500 shrink-0">Abstract</span>
                    )}
                    {model.open_source && (
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 shrink-0">OSS</span>
                    )}
                  </div>
                  <div className="text-[11px] text-zinc-500">{model.organization} · {model.date}</div>
                  {model.summary && (
                    <div className="text-xs text-zinc-400 mt-1 line-clamp-2">{model.summary}</div>
                  )}
                </div>
                <div className="ml-3 shrink-0 flex flex-col items-end gap-1">
                  {model.ratings.novelty && (
                    <span className="text-[10px] text-zinc-500">{model.ratings.novelty}</span>
                  )}
                  <svg className={`w-4 h-4 text-zinc-600 transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </button>

              {/* Expanded Review Content */}
              {isExpanded && (
                <div className="border-t border-zinc-800 px-4 py-4 bg-zinc-900/50">
                  <SimpleMarkdownCompact text={model.ai_review.content} />
                </div>
              )}
            </div>
          )
        })}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-12 text-zinc-600 text-sm">
          No reviews match your search
        </div>
      )}
    </div>
  )
}
