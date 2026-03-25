import { useEffect, useState } from 'react'
import { BENCHMARKS } from '../constants/benchmarks'

// Keys that are metadata, not numeric scores
const META_KEYS = new Set(['source', 'date_reported', 'eval_condition', 'note', 'average'])

function ScoreBar({ value, max = 100 }) {
  if (value == null) return <span className="text-zinc-600 text-xs">—</span>
  const pct = Math.min((value / max) * 100, 100)
  const color = pct >= 80 ? 'bg-emerald-500' : pct >= 50 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-medium text-white tabular-nums w-10 text-right">{value.toFixed(1)}</span>
    </div>
  )
}

function SimpleMarkdown({ text }) {
  if (!text) return null
  const lines = text.split('\n')
  const elements = []
  let inTable = false
  let tableRows = []
  let inCode = false
  let codeLines = []

  const flushTable = () => {
    if (tableRows.length === 0) return
    const headerCells = tableRows[0]
    const bodyRows = tableRows.slice(2) // skip separator
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

    if (line.startsWith('```')) {
      if (inCode) { elements.push(<pre key={`c${i}`} className="bg-zinc-800 rounded p-2 text-[11px] text-zinc-300 overflow-x-auto my-2 font-mono">{codeLines.join('\n')}</pre>); codeLines = []; inCode = false }
      else { inCode = true }
      continue
    }
    if (inCode) { codeLines.push(line); continue }

    if (line.startsWith('|') && line.includes('|')) {
      if (!inTable) inTable = true
      if (line.match(/^\|[\s-:|]+\|$/)) { tableRows.push(null); continue }
      tableRows.push(line.split('|').filter((_, idx, arr) => idx > 0 && idx < arr.length - 1))
      continue
    } else if (inTable) { inTable = false; flushTable() }

    if (line.startsWith('# ')) { elements.push(<h1 key={i} className="text-lg font-bold text-white mt-4 mb-2">{line.slice(2)}</h1>); continue }
    if (line.startsWith('## ')) { elements.push(<h2 key={i} className="text-sm font-bold text-white mt-4 mb-1.5 border-b border-zinc-800 pb-1">{line.slice(3)}</h2>); continue }
    if (line.startsWith('### ')) { elements.push(<h3 key={i} className="text-xs font-bold text-zinc-300 mt-3 mb-1">{line.slice(4)}</h3>); continue }
    if (line.startsWith('> ')) {
      const isWarning = line.includes('⚠️') || line.includes('❓')
      elements.push(<blockquote key={i} className={`border-l-2 pl-3 my-2 text-xs ${isWarning ? 'border-amber-500/50 text-amber-200/80' : 'border-purple-500/50 text-zinc-400 italic'}`}>{line.slice(2)}</blockquote>)
      continue
    }
    if (line.startsWith('- ') || line.startsWith('* ')) { elements.push(<li key={i} className="text-xs text-zinc-300 ml-4 list-disc mb-0.5">{line.slice(2)}</li>); continue }
    if (line.startsWith('---')) { elements.push(<hr key={i} className="border-zinc-800 my-3" />); continue }
    if (line.match(/^\d+\. /)) { elements.push(<li key={i} className="text-xs text-zinc-300 ml-4 list-decimal mb-0.5">{line.replace(/^\d+\. /, '')}</li>); continue }
    if (line.trim() === '') { elements.push(<div key={i} className="h-1" />); continue }
    if (line.startsWith('<!--')) continue

    elements.push(<p key={i} className="text-xs text-zinc-300 mb-1">{line.replace(/\*\*(.*?)\*\*/g, '<b>$1</b>').split('<b>').map((part, pi) => {
      if (pi === 0) return part
      const [bold, rest] = part.split('</b>')
      return <span key={pi}><strong className="text-white">{bold}</strong>{rest}</span>
    })}</p>)
  }
  if (inTable) flushTable()

  return <div>{elements}</div>
}

export default function ModelDetailModal({ model, onClose }) {
  const [modalTab, setModalTab] = useState('details')

  // Reset tab when model changes
  useEffect(() => {
    setModalTab('details')
  }, [model?.name])

  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  if (!model) return null

  const benchEntries = Object.entries(model.benchmarks || {})
  const benchCount = benchEntries.length
  const totalBenchmarks = Object.keys(BENCHMARKS).length
  const hasReview = !!model.ai_review?.content

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className={`bg-zinc-900 border border-zinc-700 rounded-2xl w-full max-h-[85vh] overflow-y-auto ${hasReview ? 'max-w-2xl' : 'max-w-lg'}`}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="sticky top-0 bg-zinc-900 border-b border-zinc-800 p-4 rounded-t-2xl z-10">
          <div className="flex justify-between items-start">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white">{model.name}</h2>
                {model.open_source && (
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-medium">OSS</span>
                )}
              </div>
              <div className="text-sm text-zinc-400 mt-0.5">{model.organization}</div>
            </div>
            <button
              onClick={onClose}
              className="text-zinc-500 hover:text-white transition-colors p-1"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          {/* Tab switcher */}
          {hasReview && (
            <div className="flex gap-1 mt-3">
              <button
                onClick={() => setModalTab('details')}
                className={`px-3 py-1 text-xs rounded-md transition-colors ${modalTab === 'details' ? 'bg-zinc-700 text-white font-medium' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                Details
              </button>
              <button
                onClick={() => setModalTab('review')}
                className={`px-3 py-1 text-xs rounded-md transition-colors flex items-center gap-1 ${modalTab === 'review' ? 'bg-purple-500/20 text-purple-300 font-medium' : 'text-zinc-500 hover:text-zinc-300'}`}
              >
                AI Review
                {model.ai_review?.verified && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" title="PDF verified" />}
              </button>
            </div>
          )}
        </div>

        {/* Review Tab */}
        {modalTab === 'review' && hasReview && (
          <div className="p-4">
            <div className="flex items-center gap-2 mb-3">
              {model.ai_review.verified ? (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">PDF Verified</span>
              ) : (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">Abstract-based</span>
              )}
            </div>
            <SimpleMarkdown text={model.ai_review.content} />
          </div>
        )}

        {/* Details Tab */}
        <div className={`p-4 space-y-4 ${modalTab !== 'details' ? 'hidden' : ''}`}>
          {/* Links */}
          <div className="flex gap-2">
            {model.paper_url && (
              <a href={model.paper_url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                </svg>
                Paper
              </a>
            )}
            {model.code_url && (
              <a href={model.code_url} target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-zinc-800 text-zinc-300 border border-zinc-700 hover:bg-zinc-700 transition-colors">
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
                Code
              </a>
            )}
          </div>

          {/* Specs grid */}
          <div className="grid grid-cols-2 gap-2">
            {[
              { label: 'Date', value: model.date },
              { label: 'Venue', value: model.venue || 'Preprint' },
              { label: 'Parameters', value: model.architecture?.parameters || '—' },
              { label: 'Inference', value: model.inference_hz ? `${model.inference_hz} Hz` : '—' },
              { label: 'Action Head', value: model.architecture?.action_head || '—' },
              { label: 'VLM Backbone', value: model.architecture?.backbone || '—' },
              { label: 'LLM', value: model.architecture?.llm || '—' },
              { label: 'Benchmark Coverage', value: `${benchCount} / ${totalBenchmarks}` },
            ].map(({ label, value }) => (
              <div key={label} className="bg-zinc-800/50 rounded-lg px-3 py-2">
                <div className="text-[10px] text-zinc-500 uppercase tracking-wider">{label}</div>
                <div className="text-xs text-white mt-0.5 truncate" title={typeof value === 'string' ? value : undefined}>{value}</div>
              </div>
            ))}
          </div>

          {/* Tags */}
          {model.tags && model.tags.length > 0 && (
            <div>
              <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5">Tags</div>
              <div className="flex flex-wrap gap-1">
                {model.tags.map(tag => (
                  <span key={tag} className="px-2 py-0.5 text-[10px] rounded bg-zinc-800 text-zinc-400 border border-zinc-700/50">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Benchmark Scores */}
          <div>
            <div className="text-[10px] text-zinc-500 uppercase tracking-wider mb-2">Benchmark Scores</div>
            {benchCount === 0 ? (
              <div className="text-sm text-zinc-600 py-3 text-center bg-zinc-800/30 rounded-lg">
                No benchmark scores available
              </div>
            ) : (
              <div className="space-y-2">
                {Object.entries(BENCHMARKS).map(([key, bench]) => {
                  const scores = model.benchmarks?.[key]
                  if (!scores) return null
                  const avg = model[bench.key]
                  return (
                    <div key={key} className="bg-zinc-800/30 rounded-lg p-2.5">
                      <div className="flex justify-between items-center mb-1.5">
                        <span className="text-xs font-semibold text-zinc-300">{bench.label}</span>
                        {avg != null && (
                          <span className="text-xs font-bold text-emerald-400">Avg: {avg.toFixed(1)}</span>
                        )}
                      </div>
                      <div className="space-y-1">
                        {Object.entries(scores).map(([suiteKey, val]) => {
                          if (suiteKey.endsWith('_avg')) return null
                          if (META_KEYS.has(suiteKey)) return null
                          if (suiteKey.endsWith('_note')) return null
                          if (typeof val !== 'number') return null
                          const label = suiteKey
                            .replace(/^(libero_|calvin_|google_robot_|robotwin_|rlbench_|robocasa_)/, '')
                            .replace(/_/g, ' ')
                          return (
                            <div key={suiteKey} className="flex items-center justify-between">
                              <span className="text-[11px] text-zinc-500 capitalize w-28 truncate">{label}</span>
                              <ScoreBar value={val} max={key === 'calvin' ? 5 : 100} />
                            </div>
                          )
                        })}
                      </div>
                      {model.eval_conditions?.[key] && (
                        <div className="mt-1 text-[10px] text-zinc-600">
                          Eval: {model.eval_conditions[key]}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          {/* Benchmarks this model is NOT in */}
          {benchCount > 0 && benchCount < totalBenchmarks && (
            <div className="text-[10px] text-zinc-600">
              Not evaluated: {Object.entries(BENCHMARKS)
                .filter(([key]) => !model.benchmarks?.[key])
                .map(([, b]) => b.label)
                .join(', ')}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
