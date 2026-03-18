import { useState } from 'react'
import LeaderboardTable from './components/LeaderboardTable'
import PerformanceChart from './components/PerformanceChart'
import ModelCompare from './components/ModelCompare'
import WeeklyDigest from './components/WeeklyDigest'
import leaderboard from './data/leaderboard.json'
import weeklyDigest from './data/weeklyDigest.json'
import modelHistory from './data/modelHistory.json'

const TABS = [
  { id: 'leaderboard', label: 'Leaderboard' },
  { id: 'trends', label: 'Trends' },
  { id: 'compare', label: 'Compare' },
  { id: 'digest', label: 'AI Analysis' },
]

export default function App() {
  const [activeTab, setActiveTab] = useState('leaderboard')

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
          <LeaderboardTable models={leaderboard.models} />
        )}
        {activeTab === 'trends' && (
          <PerformanceChart modelHistory={modelHistory} />
        )}
        {activeTab === 'compare' && (
          <ModelCompare models={leaderboard.models} />
        )}
        {activeTab === 'digest' && (
          <WeeklyDigest digest={weeklyDigest} />
        )}
      </div>

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
