import React, { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Play, Pause, Trash2, RefreshCw, AlertCircle, CheckCircle2, Clock, Bot, Layers, ShieldCheck, BookOpen, GitBranch,
} from 'lucide-react'
import { getProject, runPipeline, getTasks, pauseProject, deleteProject } from '../api/client'
import { useSSE } from '../hooks/useSSE'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'

const AGENT_NAMES = [
  'planner',
  'research',
  'verification',
  'writer',
  'grounding',
  'reviewer',
  'coherence',
  'citation',
  'figure',
]

function parseProjectContent(project) {
  if (!project?.content) return { sections: {}, metadata: {} }
  if (typeof project.content === 'object') {
    return {
      sections: project.content.sections || {},
      metadata: project.content.metadata || {},
    }
  }

  try {
    const parsed = JSON.parse(project.content)
    return {
      sections: parsed?.sections || {},
      metadata: parsed?.metadata || {},
    }
  } catch {
    return { sections: {}, metadata: {} }
  }
}

function formatSectionName(sectionKey) {
  return String(sectionKey || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function formatDuration(start, end) {
  if (!start) return '—'
  const s = new Date(start)
  const e = end ? new Date(end) : new Date()
  const secs = Math.round((e - s) / 1000)
  if (secs < 60) return `${secs}s`
  return `${Math.floor(secs / 60)}m ${secs % 60}s`
}

function ProgressBar({ tasks }) {
  const total = tasks.length
  if (total === 0) return null
  const done = tasks.filter((t) => ['completed', 'success'].includes(t.status?.toLowerCase())).length
  const pct = Math.round((done / total) * 100)
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-500">
        <span>{done} / {total} tasks completed</span>
        <span>{pct}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-primary-600 h-2 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function ProjectView() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [project, setProject] = useState(null)
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [running, setRunning] = useState(false)
  const [runMsg, setRunMsg] = useState(null)
  const [actionBusy, setActionBusy] = useState(false)

  const { events, connected } = useSSE(id)

  const fetchData = useCallback(async ({ background = false } = {}) => {
    if (background) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }
    setError(null)
    try {
      const [projRes, tasksRes] = await Promise.all([
        getProject(id),
        getTasks(id).catch(() => ({ data: [] })),
      ])
      setProject(projRes.data)
      const raw = tasksRes.data
      setTasks(Array.isArray(raw) ? raw : (raw?.tasks ?? []))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [id])

  useEffect(() => { fetchData() }, [fetchData])

  // Refresh project + tasks on terminal pipeline events; refresh tasks only on step events
  useEffect(() => {
    if (events.length === 0) return
    const latest = events[0] // events are prepended, newest first
    const isTerminal = ['pipeline_complete', 'pipeline_error', 'pipeline_paused', 'project_deleted'].includes(latest?.type)
    if (isTerminal) {
      fetchData({ background: true })
    } else {
      getTasks(id)
        .then((r) => {
          const raw = r.data
          setTasks(Array.isArray(raw) ? raw : (raw?.tasks ?? []))
        })
        .catch(() => {})
    }
  }, [events, id, fetchData])

  // Fallback polling every 8s while the pipeline is running (in case SSE events are missed)
  useEffect(() => {
    const status = (project?.status || '').toLowerCase()
    if (status !== 'running') return
    const timer = setInterval(() => fetchData({ background: true }), 8000)
    return () => clearInterval(timer)
  }, [project?.status, fetchData])

  const handleRun = async () => {
    setRunning(true)
    setRunMsg(null)
    try {
      await runPipeline(id)
      setProject((prev) => (prev ? { ...prev, status: 'running' } : prev))
      setRunMsg({ type: 'success', text: 'Pipeline started successfully! Tracking progress live.' })
      setTimeout(() => fetchData({ background: true }), 1000)
    } catch (err) {
      setRunMsg({ type: 'error', text: err.message })
    } finally {
      setRunning(false)
    }
  }

  const handlePause = async () => {
    setActionBusy(true)
    setRunMsg(null)
    try {
      await pauseProject(id)
      setRunMsg({ type: 'success', text: 'Pause requested. The pipeline will stop after the current step.' })
      await fetchData({ background: true })
    } catch (err) {
      setRunMsg({ type: 'error', text: err.message })
    } finally {
      setActionBusy(false)
    }
  }

  const handleDelete = async () => {
    const confirmed = window.confirm('Delete this project? This cannot be undone.')
    if (!confirmed) return

    setActionBusy(true)
    setRunMsg(null)
    try {
      await deleteProject(id)
      navigate('/')
    } catch (err) {
      setRunMsg({ type: 'error', text: err.message })
      setActionBusy(false)
    }
  }

  const getAgentStatus = (agentName) => {
    const agentTasks = tasks.filter(
      (t) => (t.agent_name || t.agent || '').toLowerCase() === agentName.toLowerCase()
    )
    if (agentTasks.length === 0) return 'pending'
    const last = agentTasks[agentTasks.length - 1]
    return last.status || 'pending'
  }

  const getAgentLastActivity = (agentName) => {
    const agentTasks = tasks.filter(
      (t) => (t.agent_name || t.agent || '').toLowerCase() === agentName.toLowerCase()
    )
    if (agentTasks.length === 0) return null
    const last = agentTasks[agentTasks.length - 1]
    return last.completed_at || last.started_at || null
  }

  if (loading && !project) return <div className="flex justify-center py-20"><LoadingSpinner size="lg" /></div>
  if (error) return (
    <div className="max-w-xl mx-auto mt-12 bg-red-50 border border-red-200 rounded-xl p-6 text-center">
      <AlertCircle className="mx-auto mb-2 text-red-500" size={32} />
      <p className="text-red-700 font-medium">{error}</p>
      <button onClick={fetchData} className="mt-3 text-sm text-primary-600 underline">Retry</button>
    </div>
  )

  const parsedContent = parseProjectContent(project)
  const quality = parsedContent.metadata?.quality || {}
  const qualitySummary = quality.summary || {}
  const qualitySections = quality.sections || {}
  const sectionSubheadings = parsedContent.metadata?.subheadings || {}
  const coherence = quality.coherence || null
  const topSources = parsedContent.metadata?.research?.top_sources || []

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{project?.title || project?.topic || 'Project'}</h1>
            {project?.topic && project.title !== project.topic && (
              <p className="text-gray-500 mt-1">{project.topic}</p>
            )}
            <div className="flex items-center gap-3 mt-3">
              <StatusBadge status={project?.status} />
              <span className={`flex items-center gap-1.5 text-xs ${connected ? 'text-green-600' : 'text-gray-400'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-green-500' : 'bg-gray-400'}`} />
                {connected ? 'Live' : 'Disconnected'}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => fetchData({ background: true })}
              disabled={loading || refreshing}
              className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-50"
            >
              <RefreshCw size={14} />
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
            <button
              onClick={handlePause}
              disabled={running || actionBusy || ['paused', 'completed', 'failed'].includes((project?.status || '').toLowerCase())}
              className="flex items-center gap-2 bg-amber-600 hover:bg-amber-700 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              <Pause size={14} />
              Pause
            </button>
            <button
              onClick={handleRun}
              disabled={running || actionBusy}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              {running ? <LoadingSpinner size="sm" /> : <Play size={14} />}
              {running ? 'Starting…' : 'Start Pipeline'}
            </button>
            <button
              onClick={handleDelete}
              disabled={actionBusy || (project?.status || '').toLowerCase() === 'running'}
              className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-5 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
            >
              <Trash2 size={14} />
              Delete
            </button>
          </div>
        </div>

        {refreshing && (
          <div className="mt-4 bg-blue-50 border border-blue-100 text-blue-700 text-sm rounded-lg px-3 py-2">
            Updating project data while the pipeline runs. You can keep reading this page.
          </div>
        )}

        {runMsg && (
          <div className={`mt-4 flex items-center gap-2 text-sm p-3 rounded-lg ${
            runMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
          }`}>
            {runMsg.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            {runMsg.text}
          </div>
        )}
      </div>

      {/* Progress */}
      {tasks.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Overall Progress</h2>
          <ProgressBar tasks={tasks} />
        </div>
      )}

      {(Object.keys(qualitySections).length > 0 || topSources.length > 0 || coherence) && (
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          <div className="xl:col-span-2 bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <ShieldCheck size={18} /> Quality Summary
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-5">
              <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
                <p className="text-xs uppercase tracking-wider text-gray-500 font-semibold">Average Score</p>
                <p className="mt-2 text-2xl font-bold text-gray-900">{qualitySummary.average_score ?? '—'}</p>
              </div>
              <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
                <p className="text-xs uppercase tracking-wider text-gray-500 font-semibold">Approved Sections</p>
                <p className="mt-2 text-2xl font-bold text-green-700">{qualitySummary.approved_sections ?? 0}</p>
              </div>
              <div className="rounded-lg border border-gray-100 bg-gray-50 p-4">
                <p className="text-xs uppercase tracking-wider text-gray-500 font-semibold">Flagged Sections</p>
                <p className="mt-2 text-2xl font-bold text-amber-700">{qualitySummary.flagged_sections ?? 0}</p>
              </div>
            </div>

            {coherence && (
              <div className="rounded-lg border border-gray-100 bg-gray-50 p-4 mb-5">
                <div className="flex items-start justify-between gap-3 flex-wrap">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-gray-500 font-semibold flex items-center gap-1.5">
                      <GitBranch size={12} /> Essay Coherence
                    </p>
                    <p className="mt-2 text-sm font-semibold text-gray-900">
                      Score {coherence.score ?? qualitySummary.coherence_score ?? '—'}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Topic coverage {coherence.topic_coverage ?? '—'}
                    </p>
                    {(qualitySummary.coherence_revision_rounds ?? 0) > 0 && (
                      <p className="text-xs text-gray-500 mt-1">
                        Revision rounds {qualitySummary.coherence_revision_rounds} · stop reason {qualitySummary.coherence_stop_reason || '—'}
                      </p>
                    )}
                  </div>
                  <StatusBadge
                    status={coherence.approved ? 'completed' : 'pending'}
                    className={coherence.approved ? '' : '!bg-amber-100 !text-amber-800'}
                  />
                </div>
                {coherence.issues?.length > 0 && (
                  <ul className="mt-3 space-y-1 text-xs text-amber-700">
                    {coherence.issues.slice(0, 3).map((issue, index) => (
                      <li key={`coherence-issue-${index}`}>• {issue}</li>
                    ))}
                  </ul>
                )}
                {coherence.suggestions?.length > 0 && (
                  <ul className="mt-3 space-y-1 text-xs text-blue-700">
                    {coherence.suggestions.slice(0, 2).map((item, index) => (
                      <li key={`coherence-suggestion-${index}`}>• {item}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            <div className="space-y-3">
              {Object.entries(qualitySections).map(([sectionKey, entry]) => (
                <div key={sectionKey} className="rounded-lg border border-gray-100 p-4">
                  {Array.isArray(sectionSubheadings[sectionKey]) && sectionSubheadings[sectionKey].length > 0 && (
                    <p className="text-[11px] uppercase tracking-wider text-indigo-700 font-semibold mb-2">
                      Subheadings {sectionSubheadings[sectionKey].length}
                    </p>
                  )}
                  <div className="flex items-start justify-between gap-3 flex-wrap">
                    <div>
                      <p className="font-semibold text-gray-900">{entry.title || formatSectionName(sectionKey)}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        Score {entry.score ?? '—'} · Grounding {entry.grounding_score ?? '—'} · Revisions {entry.revision_attempts ?? 0}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Evidence {entry.evidence_count ?? '—'} · Stop {entry.stop_reason || '—'}
                      </p>
                    </div>
                    <StatusBadge status={entry.approved ? 'completed' : 'pending'} className={entry.approved ? '' : '!bg-amber-100 !text-amber-800'} />
                  </div>
                  {Array.isArray(sectionSubheadings[sectionKey]) && sectionSubheadings[sectionKey].length > 0 && (
                    <ul className="mt-3 space-y-1 text-xs text-indigo-700">
                      {sectionSubheadings[sectionKey].slice(0, 2).map((sub, index) => (
                        <li key={`${sectionKey}-sub-${index}`}>• {sub?.title || `Subheading ${index + 1}`}</li>
                      ))}
                    </ul>
                  )}
                  {entry.blocking_issues?.length > 0 && (
                    <ul className="mt-3 space-y-1 text-xs text-amber-700">
                      {entry.blocking_issues.slice(0, 2).map((issue, index) => (
                        <li key={`${sectionKey}-issue-${index}`}>• {issue}</li>
                      ))}
                    </ul>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
            <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <BookOpen size={18} /> Top Sources
            </h2>
            {topSources.length === 0 ? (
              <p className="text-sm text-gray-400">No ranked sources available yet.</p>
            ) : (
              <div className="space-y-3">
                {topSources.map((source, index) => (
                  <div key={`${source.title}-${index}`} className="rounded-lg border border-gray-100 p-4 bg-gray-50">
                    <p className="text-sm font-semibold text-gray-900">{source.title}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {source.source || 'unknown'} · {source.year || 'n.d.'} · score {source.relevance_score ?? '—'}
                    </p>
                    {source.match_reasons?.length > 0 && (
                      <ul className="mt-2 space-y-1 text-xs text-gray-600">
                        {source.match_reasons.slice(0, 2).map((reason, reasonIndex) => (
                          <li key={`${source.title}-reason-${reasonIndex}`}>• {reason}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Agent Grid */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center gap-2">
          <Bot size={18} /> Agent Status
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {AGENT_NAMES.map((name) => {
            const status = getAgentStatus(name)
            const lastActivity = getAgentLastActivity(name)
            return (
              <div
                key={name}
                className="border border-gray-100 rounded-lg p-4 bg-gray-50 hover:bg-white transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-800 capitalize">{name}</span>
                  <StatusBadge status={status} />
                </div>
                {lastActivity && (
                  <p className="text-xs text-gray-400 flex items-center gap-1">
                    <Clock size={10} />
                    {new Date(lastActivity).toLocaleTimeString()}
                  </p>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Task List */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center gap-2">
          <Layers size={16} />
          <h2 className="text-lg font-semibold text-gray-900">Tasks</h2>
          <span className="text-sm text-gray-400">({tasks.length})</span>
        </div>
        {tasks.length === 0 ? (
          <div className="py-12 text-center text-gray-400 text-sm">
            No tasks yet. Start the pipeline to begin.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  <th className="px-6 py-3">Task</th>
                  <th className="px-6 py-3">Agent</th>
                  <th className="px-6 py-3">Status</th>
                  <th className="px-6 py-3">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {tasks.map((task) => (
                  <tr key={task.id || task._id || task.task_id} className="hover:bg-gray-50">
                    <td className="px-6 py-3 font-medium text-gray-800">
                      {task.name || task.task_name || task.id || '—'}
                    </td>
                    <td className="px-6 py-3 text-gray-600">
                      {task.agent_name || task.agent || '—'}
                    </td>
                    <td className="px-6 py-3">
                      <StatusBadge status={task.status} />
                    </td>
                    <td className="px-6 py-3 text-gray-500">
                      {formatDuration(task.started_at, task.completed_at)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent SSE Events */}
      {events.length > 0 && (
        <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">Live Events</h2>
          <ul className="space-y-1.5 max-h-48 overflow-y-auto scrollbar-thin">
            {events.slice(0, 20).map((ev, i) => (
              <li key={i} className="text-xs text-gray-600 bg-gray-50 rounded px-3 py-1.5 font-mono">
                {ev.raw || JSON.stringify(ev)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
