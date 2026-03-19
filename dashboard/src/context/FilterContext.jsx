import { createContext, useContext, useState, useMemo } from 'react'

const FilterContext = createContext(null)

const ACTION_HEAD_CATEGORIES = {
  'Flow Matching': 'flow matching',
  'Autoregressive': 'autoregressive',
  'Diffusion': 'diffusion',
  'MoE': 'moe',
  'FAST Tokenizer': 'fast',
}

function parseParamNumber(paramStr) {
  if (!paramStr) return null
  const match = paramStr.match(/([\d.]+)\s*[Bb]/)
  if (match) return parseFloat(match[1])
  const mMatch = paramStr.match(/([\d.]+)\s*[Mm]/)
  if (mMatch) return parseFloat(mMatch[1]) / 1000
  return null
}

function categorizeActionHead(actionHead) {
  if (!actionHead) return 'Other'
  const lower = actionHead.toLowerCase()
  for (const [category, keyword] of Object.entries(ACTION_HEAD_CATEGORIES)) {
    if (lower.includes(keyword)) return category
    if (category === 'MoE' && lower.includes('mixture')) return category
  }
  return 'Other'
}

const DEFAULT_FILTERS = {
  search: '',
  openSource: 'all',      // 'all' | 'oss' | 'closed'
  actionHead: 'all',      // 'all' | category name
  paramMin: '',
  paramMax: '',
  dateFrom: '',
  dateTo: '',
  hasBenchmarks: 'all',   // 'all' | 'yes' | 'no'
}

export function FilterProvider({ children, allModels }) {
  const [filters, setFilters] = useState(DEFAULT_FILTERS)

  const filterMeta = useMemo(() => {
    const actionHeadCounts = {}
    let ossCount = 0
    let closedCount = 0
    let withBench = 0
    let withoutBench = 0

    for (const m of allModels) {
      const cat = categorizeActionHead(m.architecture?.action_head)
      actionHeadCounts[cat] = (actionHeadCounts[cat] || 0) + 1
      if (m.open_source) ossCount++; else closedCount++
      const hasBench = m.benchmarks && Object.keys(m.benchmarks).length > 0
      if (hasBench) withBench++; else withoutBench++
    }

    return { actionHeadCounts, ossCount, closedCount, withBench, withoutBench }
  }, [allModels])

  const filteredModels = useMemo(() => {
    return allModels.filter(m => {
      // Text search
      if (filters.search) {
        const q = filters.search.toLowerCase()
        const searchable = [
          m.name, m.organization, m.architecture?.action_head,
          m.venue, ...(m.tags || [])
        ].filter(Boolean).join(' ').toLowerCase()
        if (!searchable.includes(q)) return false
      }

      // Open source filter
      if (filters.openSource === 'oss' && !m.open_source) return false
      if (filters.openSource === 'closed' && m.open_source) return false

      // Action head filter
      if (filters.actionHead !== 'all') {
        const cat = categorizeActionHead(m.architecture?.action_head)
        if (cat !== filters.actionHead) return false
      }

      // Parameter range
      const paramNum = parseParamNumber(m.architecture?.parameters)
      if (filters.paramMin && paramNum !== null && paramNum < parseFloat(filters.paramMin)) return false
      if (filters.paramMax && paramNum !== null && paramNum > parseFloat(filters.paramMax)) return false

      // Date range
      if (filters.dateFrom && m.date && m.date < filters.dateFrom) return false
      if (filters.dateTo && m.date && m.date > filters.dateTo) return false

      // Has benchmarks
      if (filters.hasBenchmarks === 'yes') {
        if (!m.benchmarks || Object.keys(m.benchmarks).length === 0) return false
      }
      if (filters.hasBenchmarks === 'no') {
        if (m.benchmarks && Object.keys(m.benchmarks).length > 0) return false
      }

      return true
    })
  }, [allModels, filters])

  const activeFilterCount = useMemo(() => {
    let count = 0
    if (filters.search) count++
    if (filters.openSource !== 'all') count++
    if (filters.actionHead !== 'all') count++
    if (filters.paramMin || filters.paramMax) count++
    if (filters.dateFrom || filters.dateTo) count++
    if (filters.hasBenchmarks !== 'all') count++
    return count
  }, [filters])

  const resetFilters = () => setFilters(DEFAULT_FILTERS)

  const value = {
    filters,
    setFilters,
    filteredModels,
    allModels,
    filterMeta,
    activeFilterCount,
    resetFilters,
    categorizeActionHead,
  }

  return (
    <FilterContext.Provider value={value}>
      {children}
    </FilterContext.Provider>
  )
}

export function useFilters() {
  const ctx = useContext(FilterContext)
  if (!ctx) throw new Error('useFilters must be used within FilterProvider')
  return ctx
}

export { DEFAULT_FILTERS, ACTION_HEAD_CATEGORIES, categorizeActionHead, parseParamNumber }
