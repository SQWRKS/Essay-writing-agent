import React, { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { ChevronDown, ChevronRight, Filter, RefreshCw, AlertCircle } from 'lucide-react'
import { getTasks } from '../api/client'
import { useSSE } from '../hooks/useSSE'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'

function formatDate(d) {
  if (!d) return '—'
  try { return new Date(d).toLocaleString() } catch { return d }
}

function formatDuration(start, end) {
  if (!start) return '—'
  const s = new Date(start)
  const e = end ? new Date(end) : new Date()
  const ms = e - s
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`
}

const AGENT_OPTIONS = [
  'All Agents',
  'ResearchAgent',
  'OutlineAgent',
  'WriterAgent',
  'CitationAgent',
  'ReviewAgent',
  'FormatterAgent',
  'QualityAgent',
]

const STATUS_OPTIONS = ['All Statuses', 'pending', 'running', 'completed', 'failed']

export default function AgentMonitor() {
  const { id } = useParams()
  const [tasks, setTasks] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expanded, setExpanded] = useState({})
  const [agentFilter, setAgentFilter] = useState('All Agents')
  const [statusFilter, setStatusFilter] = useState('All Statuses')

  const { events, connected } = useSSE(id)

  const fetchTasks = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getTasks(id)
      const raw = res.data
      setTasks(Array.isArray(raw) ? raw : (raw?.tasks ?? []))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [id])

  useEffect(() => { fetchTasks() }, [fetchTasks])

  useEffect(() => {
    if (events.length > 0) fetchTasks()
  }, [events, fetchTasks])

  const toggleExpand = (taskId) =>
    setExpanded((prev) => ({ ...prev, [taskId]: !prev[taskId] }))

  const filtered = tasks.filter((t) => {
    const agentMatch =
      agentFilter === 'All Agents' ||
      (t.agent_name || t.agent || '').toLowerCase() === agentFilter.toLowerCase()
    const statusMatch =
      statusFilter === 'All Statuses' ||
      (t.status || '').toLowerCase() === statusFilter.toLowerCase()
    return agentMatch && statusMatch
  })

  if (loading) return <div className="flex justify-center py-20"><LoadingSpinner size="lg" /></div>
  if (error) return (
    <div className="max-w-xl mx-auto mt-12 bg-red-50 border border-red-200 rounded-xl p-6 text-center">
      <AlertCircle className="mx-auto mb-2 text-red-500" size={32} />
      <p className="text-red-700">{error}</p>
      <button onClick={fetchTasks} className="mt-3 text-sm text-primary-600 underline">Retry</button>
    </div>
  )

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Agent Monitor</h1>
          <p className="text-gray-500 text-sm mt-0.5">
            {tasks.length} tasks total ·{' '}
            <span className={connected ? 'text-green-600' : 'text-gray-400'}>
              {connected ? '● Live' : '○ Disconnected'}
            </span>
          </p>
        </div>
        <button
          onClick={fetchTasks}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-50"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-4 flex items-center gap-4 flex-wrap">
        <Filter size={16} className="text-gray-400" />
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-500">Agent:</label>
          <select
            value={agentFilter}
            onChange={(e) => setAgentFilter(e.target.value)}
            className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {AGENT_OPTIONS.map((o) => <option key={o}>{o}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-xs font-medium text-gray-500">Status:</label>
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-sm border border-gray-200 rounded-lg px-2 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            {STATUS_OPTIONS.map((o) => <option key={o}>{o}</option>)}
          </select>
        </div>
        <span className="text-xs text-gray-400 ml-auto">{filtered.length} matching</span>
      </div>

      {/* Tasks Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {filtered.length === 0 ? (
          <div className="py-16 text-center text-gray-400 text-sm">No tasks match the current filters.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  <th className="px-4 py-3 w-8" />
                  <th className="px-4 py-3">Task Name</th>
                  <th className="px-4 py-3">Agent</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Started</th>
                  <th className="px-4 py-3">Completed</th>
                  <th className="px-4 py-3">Duration</th>
                  <th className="px-4 py-3">Depends On</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filtered.map((task) => {
                  const key = task.id || task._id || task.task_id
                  const isExpanded = expanded[key]
                  const hasOutput = task.output || task.result || task.error_message

                  return (
                    <React.Fragment key={key}>
                      <tr className="hover:bg-gray-50 transition-colors">
                        <td className="px-4 py-3">
                          {hasOutput && (
                            <button
                              onClick={() => toggleExpand(key)}
                              className="text-gray-400 hover:text-gray-700"
                            >
                              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                            </button>
                          )}
                        </td>
                        <td className="px-4 py-3 font-medium text-gray-800">
                          {task.name || task.task_name || key || '—'}
                        </td>
                        <td className="px-4 py-3 text-gray-600">
                          {task.agent_name || task.agent || '—'}
                        </td>
                        <td className="px-4 py-3">
                          <StatusBadge status={task.status} />
                        </td>
                        <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                          {formatDate(task.started_at)}
                        </td>
                        <td className="px-4 py-3 text-gray-500 whitespace-nowrap">
                          {formatDate(task.completed_at)}
                        </td>
                        <td className="px-4 py-3 text-gray-500">
                          {formatDuration(task.started_at, task.completed_at)}
                        </td>
                        <td className="px-4 py-3 text-gray-400 text-xs">
                          {task.depends_on
                            ? (Array.isArray(task.depends_on)
                                ? task.depends_on.join(', ')
                                : task.depends_on)
                            : '—'}
                        </td>
                      </tr>
                      {isExpanded && hasOutput && (
                        <tr className="bg-gray-50">
                          <td colSpan={8} className="px-6 py-4">
                            <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                              <pre className="text-xs text-green-300 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                                {JSON.stringify(task.output || task.result || { error: task.error_message }, null, 2)}
                              </pre>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
