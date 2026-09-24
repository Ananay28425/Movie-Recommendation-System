// Purpose: render the one-page movie discovery and feedback experience.
import React, { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'

async function api(path, options) {
  // Send an API request and turn server errors into readable UI messages.
  const response = await fetch(path, options)
  const data = await response.json()
  if (!response.ok) {
    const detail = data.detail
    throw new Error(typeof detail === 'string' ? detail : 'Request failed. Check your input and try again.')
  }
  return data
}

function App() {
  // Hold the state needed to render profiles, rankings, and feedback.
  const [input, setInput] = useState('54')
  const [activeUser, setActiveUser] = useState(null)
  const [result, setResult] = useState(null)
  const [previous, setPrevious] = useState({})
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState('connecting')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')

  async function load(userId, oldResult = null) {
    // Fetch the ranked list and remember earlier scores and positions.
    setBusy(true)
    setError('')
    try {
      const data = await api(`/api/recommendations/${userId}?limit=6`)
      setPrevious(oldResult ? Object.fromEntries(oldResult.recommendations.map((movie, index) => [movie.movie_id, { score: movie.final_score, rank: index }])) : {})
      setResult(data)
      setActiveUser(userId)
      setStatus('online')
    } catch (err) {
      setError(err.message)
      if (err instanceof TypeError) setStatus('offline')
      if (!oldResult) {
        setResult(null)
        setActiveUser(null)
      }
    } finally {
      setBusy(false)
    }
  }

  // Check the API and load the default profile when the page first mounts.
  useEffect(() => {
    api('/api/health').then(() => setStatus('online')).catch(() => setStatus('offline'))
    load(54)
  }, [])

  // Clear the feedback confirmation after it has been visible briefly.
  useEffect(() => {
    if (!notice) return
    const timer = window.setTimeout(() => setNotice(''), 4500)
    return () => window.clearTimeout(timer)
  }, [notice])

  function submit(event) {
    // Validate the selected user ID before loading that profile.
    event.preventDefault()
    const id = Number(input)
    if (!Number.isInteger(id) || id < 1) {
      setError('Enter a positive whole number for the MovieLens user ID.')
      return
    }
    setNotice('')
    setEvents([])
    load(id)
  }

  async function feedback(movie, signal) {
    // Save the signal and immediately reload recommendations.
    setBusy(true)
    setError('')
    try {
      const saved = await api('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: activeUser, movie_id: movie.movie_id, signal }),
      })
      setEvents(current => [{ title: saved.title, signal }, ...current].slice(0, 4))
      const genres = movie.genres.split('|').filter(genre => genre !== 'Unknown').slice(0, 2)
      setNotice(`Learned from feedback: ${genres.map(genre => `${genre} ${signal === 'like' ? '↑' : '↓'}`).join('  ·  ') || saved.title}`)
      await load(activeUser, result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const profile = result?.profile
  return (
    <main className="shell">
      <header className="masthead">
        <div className="brand"><span className="brand-mark">A//M</span><span className="eyebrow">THE REC ROOM / 001</span></div>
        <span className={`model-status ${status}`}><span className="status-dot" /> ENGINE {status.toUpperCase()}</span>
      </header>

      <section className="hero">
        <div>
          <span className="overline">A BETTER WATCHLIST, ONE CLICK AT A TIME.</span>
          <h1>ADAPTIVE<span className="slashes">//</span><br />MOVIES<span className="period">.</span></h1>
          <p>Hybrid Recommendation Engine</p>
        </div>
        <div className="hero-stamp" aria-hidden="true">YOUR<br />TASTE<br /><span>↗</span></div>
      </section>

      <section className="controls" aria-label="Load a user's recommendations">
        <form onSubmit={submit} className="user-form">
          <label htmlFor="user-id">01 / CHOOSE YOUR PROFILE</label>
          <div className="input-row">
            <span className="input-prefix">USER #</span>
            <input id="user-id" type="number" min="1" step="1" value={input} onChange={event => setInput(event.target.value)} />
            <button type="submit" disabled={busy}>LOAD RECS <span aria-hidden="true">↗</span></button>
          </div>
          <small>Try a MovieLens user ID from 1–943. Each user starts with their existing ratings.</small>
        </form>
        <div className="formula" aria-label="How ranking works"><span>THE ENGINE</span><strong>CONTENT + NEIGHBORS + FOREST</strong><span className="formula-arrow">↓</span><strong>BASE SCORE + YOUR FEEDBACK</strong></div>
      </section>

      {error && <p className="alert error" role="alert">{error}</p>}
      {notice && <p className="alert learned" role="status">{notice} <span>RE-RANKED ↗</span></p>}

      <div className="main-layout">
        <aside className="sidebar">
          <section className="taste panel">
            <div className="panel-header"><span>02 / YOUR TASTE</span><span>↘</span></div>
            <h2>THE PROFILE<span className="period">.</span></h2>
            <div className="counters"><div><strong>{profile?.likes ?? '—'}</strong><span>LIKES</span></div><div><strong>{profile?.dislikes ?? '—'}</strong><span>REJECTED</span></div></div>
            <p className="section-label">TOP PREFERRED GENRES</p>
            <div className="genre-list">{profile?.top_genres?.length ? profile.top_genres.map(genre => <span key={genre}>{genre}</span>) : <em>Like a movie to start shaping your taste.</em>}</div>
          </section>
          <section className="recent panel"><div className="panel-header"><span>RECENT SIGNALS</span><span>↙</span></div>{events.length ? events.map((event, index) => <div className="event" key={`${event.title}-${index}`}><b className={event.signal}>{event.signal === 'like' ? '+' : '−'}</b><span>{event.title}</span></div>) : <p>Your feedback will show here as you go.</p>}</section>
          <p className="sidebar-note">LIVE FEEDBACK / IN MEMORY ONLY<br />RESTART THE API TO RESET IT.</p>
        </aside>

        <section className="recommendations" aria-label="Movie recommendations">
          <div className="list-heading"><div><span className="section-label">03 / THE SHORTLIST</span><h2>FOR YOU<span className="period">.</span></h2></div><span className="issue-number">{activeUser ? `USER ${activeUser} / 06 PICKS` : 'AWAITING PROFILE'}</span></div>
          {busy && !result && <p className="empty">Training the hybrid model and loading your picks…</p>}
          {!busy && result && !result.recommendations.length && <p className="empty">No unseen movies left for this user.</p>}
          <div className="movie-grid">
            {result?.recommendations.map((movie, index) => {
              const change = previous[movie.movie_id]
              const scoreDiff = change ? movie.final_score - change.score : 0
              return <article className="movie-card" key={movie.movie_id}>
                <div className="card-top"><span>NO. {String(index + 1).padStart(2, '0')} / ID {movie.movie_id}</span>{change && change.rank !== index && <span className="shift">{change.rank > index ? '↑' : '↓'} {Math.abs(change.rank - index)} PLACES</span>}{!change && Object.keys(previous).length > 0 && <span className="shift">NEW PICK</span>}</div>
                <h3>{movie.title}</h3>
                <p className="genres">{movie.genres.replaceAll('|', ' / ') || 'UNCATEGORIZED'}</p>
                <div className="scoreboard"><div><span>BASE MODEL</span><strong>{movie.base_score.toFixed(2)}</strong></div><div><span>FEEDBACK ADJUSTMENT</span><strong className={movie.personalization_contribution < 0 ? 'negative' : 'positive'}>{movie.personalization_contribution >= 0 ? '+' : ''}{movie.personalization_contribution.toFixed(2)}</strong></div><div className="final-row"><span>FINAL SCORE</span><strong>{movie.final_score.toFixed(2)} <small>/ 5</small></strong></div></div>
                {change && Math.abs(scoreDiff) >= 0.005 && <p className="score-change">{scoreDiff > 0 ? '↑' : '↓'} {Math.abs(scoreDiff).toFixed(2)} since your last signal</p>}
                <p className="explanation">{movie.explanation}.</p>
                <div className="actions"><button type="button" className="like" onClick={() => feedback(movie, 'like')} disabled={busy}>+ LIKE</button><button type="button" className="reject" onClick={() => feedback(movie, 'reject')} disabled={busy}>− REJECT</button></div>
              </article>
            })}
          </div>
          {busy && result && <p className="loading" role="status">UPDATING YOUR SHORTLIST…</p>}
        </section>
      </div>
      <footer><span>ADAPTIVE//MOVIES</span><span>BUILT WITH FASTAPI × REACT × MOVIELENS 100K</span><span>© THE REC ROOM</span></footer>
    </main>
  )
}

createRoot(document.getElementById('root')).render(<App />)
