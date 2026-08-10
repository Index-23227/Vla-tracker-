import { useEffect, useState } from 'react'

// Review bodies are ~92% of leaderboard.json, so build_leaderboard.py strips them
// from the bundled copy and emits them as fetched assets instead:
//   /reviews.json        full name→markdown map (Reviews tab needs it for search)
//   /reviews/{slug}.md   one model (detail modal opens a single review)
// Keep reviewSlug() in sync with review_slug() in scripts/build_leaderboard.py.

const base = import.meta.env.BASE_URL || '/'

export function reviewSlug(name) {
  return String(name).replace(/[^A-Za-z0-9._-]/g, '_')
}

let allReviewsPromise = null

export function loadAllReviews() {
  if (!allReviewsPromise) {
    allReviewsPromise = fetch(`${base}reviews.json`)
      .then(r => {
        if (!r.ok) throw new Error(`reviews.json: ${r.status}`)
        return r.json()
      })
      .catch(err => {
        allReviewsPromise = null // let a later mount retry
        throw err
      })
  }
  return allReviewsPromise
}

const oneReviewCache = new Map()

export function loadReview(name) {
  if (!oneReviewCache.has(name)) {
    // If the full map is already in flight or resolved, read from it rather
    // than issuing a second request for the same text.
    const promise = allReviewsPromise
      ? allReviewsPromise.then(all => all[name] ?? '')
      : fetch(`${base}reviews/${reviewSlug(name)}.md`)
          .then(r => {
            if (!r.ok) throw new Error(`review ${name}: ${r.status}`)
            return r.text()
          })
          .catch(err => {
            oneReviewCache.delete(name)
            throw err
          })
    oneReviewCache.set(name, promise)
  }
  return oneReviewCache.get(name)
}

function useAsync(loader, deps, enabled = true) {
  const [state, setState] = useState({ data: null, loading: enabled, error: null })

  useEffect(() => {
    if (!enabled) {
      setState({ data: null, loading: false, error: null })
      return
    }
    let alive = true
    setState(s => ({ ...s, loading: true, error: null }))
    loader()
      .then(data => alive && setState({ data, loading: false, error: null }))
      .catch(error => alive && setState({ data: null, loading: false, error }))
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}

/** All review bodies as { modelName: markdown }. */
export function useAllReviews() {
  const { data, loading, error } = useAsync(loadAllReviews, [])
  return { reviews: data ?? {}, loading, error }
}

/** Markdown for a single model; pass a falsy name to skip fetching. */
export function useReview(name) {
  const { data, loading, error } = useAsync(() => loadReview(name), [name], !!name)
  return { content: data ?? '', loading, error }
}
