import { useState, useCallback } from 'react'
import { useFilters, ACTION_HEAD_CATEGORIES } from '../context/FilterContext'

function downloadFile(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export default function GlobalFilterBar() {
  const {
    filters, setFilters, filteredModels, allModels,
    filterMeta, activeFilterCount, resetFilters,
  } = useFilters()

  const [expanded, setExpanded] = useState(false)
  const [showExport, setShowExport] = useState(false)

  const update = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const toggleBtn = (currentVal, targetVal) =>
    currentVal === targetVal ? 'all' : targetVal

  const exportJSON = useCallback(() => {
    const data = { exported_at: new Date().toISOString(), count: filteredModels.length, models: filteredModels }
    downloadFile(JSON.stringify(data, null, 2), 'vla-tracker-export.json', 'application/json')
    setShowExport(false)
  }, [filteredModels])

  const exportCSV = useCallback(() => {
    const benchKeys = ['libero_avg', 'calvin_avg', 'simpler_avg', 'robotwin_v1_avg', 'robotwin_v2_avg', 'rlbench_avg', 'maniskill_avg', 'vlabench_avg', 'robocasa_avg']
    const headers = ['Name', 'Organization', 'Date', 'Venue', 'Open Source', 'Parameters', 'Action Head', 'Inference Hz', ...benchKeys]
    const rows = filteredModels.map(m => [
      m.name, m.organization, m.date, m.venue || '', m.open_source ? 'Yes' : 'No',
      m.architecture?.parameters || '', m.architecture?.action_head || '', m.inference_hz || '',
      ...benchKeys.map(k => m[k] ?? ''),
    ])
    const csv = [headers, ...rows].map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n')
    downloadFile(csv, 'vla-tracker-export.csv', 'text/csv')
    setShowExport(false)
  }, [filteredModels])

  return (
    <div className="mb-5">
      {/* Search + toggle row */}
      <div className="flex gap-2 items-center">
        <div className="relative flex-1">
          <input
            type="text"
            value={filters.search}
            onChange={e => update('search', e.target.value)}
            placeholder="Search models, orgs, tags..."
            className="w-full px-3 py-2 pl-8 text-sm bg-zinc-900 border border-zinc-700 rounded-lg text-white placeholder:text-zinc-500 focus:outline-none focus:border-zinc-500 transition-colors"
          />
          <svg className="absolute left-2.5 top-2.5 w-3.5 h-3.5 text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          {filters.search && (
            <button
              onClick={() => update('search', '')}
              className="absolute right-2.5 top-2 text-zinc-500 hover:text-zinc-300 text-sm"
            >
              ✕
            </button>
          )}
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className={`px-3 py-2 text-xs rounded-lg border transition-all flex items-center gap-1.5 ${
            expanded || activeFilterCount > 0
              ? 'border-purple-500/50 bg-purple-500/10 text-purple-300'
              : 'border-zinc-700 text-zinc-400 hover:border-zinc-500'
          }`}
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
          </svg>
          Filters
          {activeFilterCount > 0 && (
            <span className="inline-flex items-center justify-center w-4 h-4 text-[10px] font-bold rounded-full bg-purple-500 text-white">
              {activeFilterCount}
            </span>
          )}
        </button>

        {/* Export dropdown */}
        <div className="relative">
          <button
            onClick={() => setShowExport(!showExport)}
            className="px-3 py-2 text-xs rounded-lg border border-zinc-700 text-zinc-400 hover:border-zinc-500 transition-all flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export
          </button>
          {showExport && (
            <div className="absolute right-0 top-10 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-10 overflow-hidden">
              <button onClick={exportJSON} className="block w-full text-left px-4 py-2 text-xs text-zinc-300 hover:bg-zinc-700 transition-colors">
                Download JSON ({filteredModels.length} models)
              </button>
              <button onClick={exportCSV} className="block w-full text-left px-4 py-2 text-xs text-zinc-300 hover:bg-zinc-700 transition-colors border-t border-zinc-700">
                Download CSV ({filteredModels.length} models)
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Filter panel */}
      {expanded && (
        <div className="mt-2 p-3 bg-zinc-900/80 border border-zinc-800 rounded-xl space-y-3">
          {/* Row 1: Quick toggles */}
          <div className="flex flex-wrap gap-2">
            <span className="text-[10px] text-zinc-500 uppercase tracking-wider self-center mr-1">Source</span>
            <button
              onClick={() => update('openSource', toggleBtn(filters.openSource, 'oss'))}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                filters.openSource === 'oss'
                  ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300 font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              Open Source ({filterMeta.ossCount})
            </button>
            <button
              onClick={() => update('openSource', toggleBtn(filters.openSource, 'closed'))}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                filters.openSource === 'closed'
                  ? 'border-red-500/50 bg-red-500/10 text-red-300 font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              Closed ({filterMeta.closedCount})
            </button>

            <span className="text-zinc-700 mx-1">|</span>

            <span className="text-[10px] text-zinc-500 uppercase tracking-wider self-center mr-1">Data</span>
            <button
              onClick={() => update('hasBenchmarks', toggleBtn(filters.hasBenchmarks, 'yes'))}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                filters.hasBenchmarks === 'yes'
                  ? 'border-blue-500/50 bg-blue-500/10 text-blue-300 font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              With Scores ({filterMeta.withBench})
            </button>
            <button
              onClick={() => update('hasBenchmarks', toggleBtn(filters.hasBenchmarks, 'no'))}
              className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                filters.hasBenchmarks === 'no'
                  ? 'border-amber-500/50 bg-amber-500/10 text-amber-300 font-semibold'
                  : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
              }`}
            >
              No Scores ({filterMeta.withoutBench})
            </button>
          </div>

          {/* Row 2: Action Head */}
          <div>
            <span className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5 block">Action Head</span>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(ACTION_HEAD_CATEGORIES).map(([cat]) => {
                const count = filterMeta.actionHeadCounts[cat] || 0
                if (count === 0) return null
                return (
                  <button
                    key={cat}
                    onClick={() => update('actionHead', toggleBtn(filters.actionHead, cat))}
                    className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                      filters.actionHead === cat
                        ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                        : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
                    }`}
                  >
                    {cat} ({count})
                  </button>
                )
              })}
              {(filterMeta.actionHeadCounts['Other'] || 0) > 0 && (
                <button
                  onClick={() => update('actionHead', toggleBtn(filters.actionHead, 'Other'))}
                  className={`px-2.5 py-1 text-[11px] rounded-md border transition-all ${
                    filters.actionHead === 'Other'
                      ? 'border-purple-500/50 bg-purple-500/10 text-purple-300 font-semibold'
                      : 'border-zinc-700/50 text-zinc-500 hover:border-zinc-600'
                  }`}
                >
                  Other ({filterMeta.actionHeadCounts['Other']})
                </button>
              )}
            </div>
          </div>

          {/* Row 3: Parameters + Date range */}
          <div className="flex flex-wrap gap-4">
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5 block">Parameters (B)</span>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  value={filters.paramMin}
                  onChange={e => update('paramMin', e.target.value)}
                  placeholder="Min"
                  step="0.1"
                  className="w-20 px-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded text-white placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                />
                <span className="text-zinc-600 text-xs">—</span>
                <input
                  type="number"
                  value={filters.paramMax}
                  onChange={e => update('paramMax', e.target.value)}
                  placeholder="Max"
                  step="0.1"
                  className="w-20 px-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded text-white placeholder:text-zinc-600 focus:outline-none focus:border-zinc-500"
                />
              </div>
            </div>

            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wider mb-1.5 block">Date Range</span>
              <div className="flex items-center gap-1.5">
                <input
                  type="month"
                  value={filters.dateFrom ? filters.dateFrom.slice(0, 7) : ''}
                  onChange={e => update('dateFrom', e.target.value ? e.target.value + '-01' : '')}
                  className="px-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-zinc-500 [color-scheme:dark]"
                />
                <span className="text-zinc-600 text-xs">—</span>
                <input
                  type="month"
                  value={filters.dateTo ? filters.dateTo.slice(0, 7) : ''}
                  onChange={e => update('dateTo', e.target.value ? e.target.value + '-31' : '')}
                  className="px-2 py-1 text-xs bg-zinc-800 border border-zinc-700 rounded text-white focus:outline-none focus:border-zinc-500 [color-scheme:dark]"
                />
              </div>
            </div>
          </div>

          {/* Reset + count */}
          <div className="flex justify-between items-center pt-1 border-t border-zinc-800">
            <span className="text-[11px] text-zinc-500">
              Showing <span className="text-white font-semibold">{filteredModels.length}</span> of {allModels.length} models
            </span>
            {activeFilterCount > 0 && (
              <button
                onClick={resetFilters}
                className="text-[11px] text-zinc-400 hover:text-white transition-colors"
              >
                Clear all filters
              </button>
            )}
          </div>
        </div>
      )}

      {/* Active filter summary (when collapsed) */}
      {!expanded && activeFilterCount > 0 && (
        <div className="mt-1.5 flex items-center gap-2 text-[11px] text-zinc-500">
          <span>
            Showing <span className="text-white font-medium">{filteredModels.length}</span> of {allModels.length} models
          </span>
          <button
            onClick={resetFilters}
            className="text-zinc-400 hover:text-white transition-colors underline"
          >
            Clear
          </button>
        </div>
      )}
    </div>
  )
}
