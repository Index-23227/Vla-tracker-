import { useState, lazy, Suspense } from 'react'
import { FilterProvider, useFilters } from './context/FilterContext'
import ErrorBoundary from './components/ErrorBoundary'
import GlobalFilterBar from './components/GlobalFilterBar'
import LeaderboardTable from './components/LeaderboardTable'
import ModelDetailModal from './components/ModelDetailModal'
import leaderboard from './data/leaderboard.json'
import weeklyDigest from './data/weeklyDigest.json'

// Lazy-load heavy tabs to reduce initial bundle size
const ModelCompare = lazy(() => import('./components/ModelCompare'))
const AnalysisDashboard = lazy(() => import('./components/AnalysisDashboard'))
const ModelLineage = lazy(() => import('./components/ModelLineage'))
const EfficiencyRanking = lazy(() => import('./components/EfficiencyRanking'))
const RealWorldBenchmark = lazy(() => import('./components/RealWorldBenchmark'))
const CoverageHeatmap = lazy(() => import('./components/CoverageHeatmap'))
const ArchitectureExplorer = lazy(() => import('./components/ArchitectureExplorer'))

function TabLoading() {
  return (
    <div className="flex items-center justify-center py-20">
      <div className="text-sm text-zinc-500">Loading...</div>
    </div>
  )
}

const TABS = [
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'compare', label: 'Compare' },
  { id: 'lineage', label: 'Lineage' },
  { id: 'efficiency', label: 'Efficiency' },
  { id: 'realworld', label: 'Real-World' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'coverage', label: 'Coverage' },
  { id: 'analysis', label: 'Analysis' },
]

function AppContent() {
  const [activeTab, setActiveTab] = useState('leaderboard')
  const [selectedModel, setSelectedModel] = useState(null)
  const { filteredModels } = useFilters()

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-5">
        <div className="flex items-center gap-2.5 mb-1">
          <span className="text-xl">🔬</span>
          <h1 className="text-2xl font-bold text-white">VLA-Tracker</h1>
          <span className="inline-block px-2 py-0.5 text-[11px] font-medium rounded-md bg-emerald-500/10 text-emerald-400">
            LIVE
          </span>
        </div>
        <p className="text-sm text-zinc-400">
          AI-powered benchmark tracking for Vision-Language-Action models · Updated weekly
        </p>
      </div>

      {/* Weekly headline banner */}
      <div className="bg-blue-500/5 border border-blue-500/20 rounded-xl px-4 py-3 mb-5">
        <div className="text-xs font-semibold text-blue-400 mb-1">
          📡 {weeklyDigest.week_display}
        </div>
        <div className="text-sm text-white">{weeklyDigest.headline}</div>
      </div>

      {/* Global Filter Bar */}
      <GlobalFilterBar />

      {/* Tab navigation */}
      <div className="flex gap-1.5 mb-5 flex-wrap">
        {TABS.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-4 py-2 text-sm rounded-lg border transition-all ${
              activeTab === tab.id
                ? 'border-zinc-600 bg-zinc-800 text-white font-semibold'
                : 'border-transparent text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="min-h-[400px]">
        {activeTab === 'leaderboard' && (
          <LeaderboardTable models={filteredModels} onModelClick={setSelectedModel} />
        )}
        <Suspense fallback={<TabLoading />}>
          {activeTab === 'compare' && (
            <ModelCompare models={filteredModels} />
          )}
          {activeTab === 'lineage' && (
            <ModelLineage models={filteredModels} />
          )}
          {activeTab === 'efficiency' && (
            <EfficiencyRanking models={filteredModels} />
          )}
          {activeTab === 'realworld' && (
            <RealWorldBenchmark models={filteredModels} />
          )}
          {activeTab === 'architecture' && (
            <ArchitectureExplorer models={filteredModels} />
          )}
          {activeTab === 'coverage' && (
            <CoverageHeatmap models={filteredModels} />
          )}
          {activeTab === 'analysis' && (
            <AnalysisDashboard models={filteredModels} />
          )}
        </Suspense>
      </div>

      {/* Model Detail Modal */}
      {selectedModel && (
        <ModelDetailModal
          model={selectedModel}
          onClose={() => setSelectedModel(null)}
        />
      )}

      {/* Footer */}
      <div className="mt-8 p-3 bg-zinc-900 rounded-xl flex justify-between items-center text-[11px] text-zinc-500">
        <span>
          {leaderboard.num_models} models · {leaderboard.benchmarks_available.length} benchmarks · Updated {leaderboard.generated_at.split('T')[0]}
        </span>
        <a
          href="https://github.com/HyeongjinKim/Vla-tracker-"
          target="_blank"
          rel="noopener noreferrer"
          className="text-zinc-400 hover:text-white transition-colors"
        >
          ⭐ Star on GitHub
        </a>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <FilterProvider allModels={leaderboard.models}>
        <AppContent />
      </FilterProvider>
    </ErrorBoundary>
  )
}
