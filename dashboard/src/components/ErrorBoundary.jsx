import { Component } from 'react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="max-w-xl mx-auto mt-20 p-6 border border-red-500/30 bg-red-500/5 rounded-xl text-center">
          <div className="text-2xl mb-3">Something went wrong</div>
          <p className="text-sm text-zinc-400 mb-4">
            The dashboard encountered an error. This usually means the data file failed to load or is malformed.
          </p>
          <pre className="text-xs text-red-400 bg-zinc-900 rounded-lg p-3 mb-4 overflow-auto text-left">
            {this.state.error?.message || 'Unknown error'}
          </pre>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 text-sm rounded-lg border border-zinc-600 bg-zinc-800 text-white hover:bg-zinc-700 transition-colors"
          >
            Reload Page
          </button>
        </div>
      )
    }

    return this.props.children
  }
}
