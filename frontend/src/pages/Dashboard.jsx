import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { PlusCircle, FolderOpen, AlertCircle, RefreshCw, BookOpen } from 'lucide-react'
import { useProjects } from '../hooks/useProjects'
import StatusBadge from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'

function formatDate(dateStr) {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric', month: 'short', day: 'numeric',
    })
  } catch {
    return dateStr
  }
}

export default function Dashboard() {
  const { projects, loading, error, refetch, createProject } = useProjects()
  const navigate = useNavigate()

  const [topic, setTopic] = useState('')
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState(null)

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!topic.trim()) return
    setCreating(true)
    setCreateError(null)
    try {
      const title = topic.trim().slice(0, 80)
      const project = await createProject(title, topic.trim())
      setTopic('')
      navigate(`/projects/${project.id || project._id}`)
    } catch (err) {
      setCreateError(err.message)
    } finally {
      setCreating(false)
    }
  }

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      {/* Hero */}
      <div className="bg-gradient-to-br from-primary-600 to-primary-900 rounded-2xl p-8 text-white shadow-lg">
        <div className="flex items-center gap-3 mb-3">
          <BookOpen size={32} />
          <h1 className="text-3xl font-bold">AI Academic Writing System</h1>
        </div>
        <p className="text-primary-50 text-lg max-w-2xl">
          Leverage multi-agent AI to research, outline, write, and refine academic essays
          automatically. Start by entering your essay topic below.
        </p>
      </div>

      {/* Create Project */}
      <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Create New Project</h2>
        <form onSubmit={handleCreate} className="flex gap-3">
          <input
            type="text"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="Enter your essay topic…"
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={creating}
          />
          <button
            type="submit"
            disabled={creating || !topic.trim()}
            className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? <LoadingSpinner size="sm" /> : <PlusCircle size={16} />}
            {creating ? 'Creating…' : 'Create Project'}
          </button>
        </form>
        {createError && (
          <p className="mt-3 text-sm text-red-600 flex items-center gap-1.5">
            <AlertCircle size={14} /> {createError}
          </p>
        )}
      </div>

      {/* Project List */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <h2 className="text-lg font-semibold text-gray-900">
            Your Projects
            {!loading && (
              <span className="ml-2 text-sm font-normal text-gray-400">
                ({projects.length})
              </span>
            )}
          </h2>
          <button
            onClick={refetch}
            disabled={loading}
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 transition-colors disabled:opacity-40"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="flex justify-center py-16">
            <LoadingSpinner size="lg" />
          </div>
        ) : error ? (
          <div className="flex flex-col items-center gap-3 py-16 text-center">
            <AlertCircle size={32} className="text-red-400" />
            <p className="text-red-600 font-medium">{error}</p>
            <button
              onClick={refetch}
              className="text-sm text-primary-600 hover:text-primary-800 underline"
            >
              Try again
            </button>
          </div>
        ) : projects.length === 0 ? (
          <div className="flex flex-col items-center gap-3 py-16 text-center text-gray-400">
            <FolderOpen size={40} />
            <p className="font-medium">No projects yet</p>
            <p className="text-sm">Create your first project above to get started.</p>
          </div>
        ) : (
          <ul className="divide-y divide-gray-100">
            {projects.map((p) => (
              <li
                key={p.id || p._id}
                className="flex items-center justify-between px-6 py-4 hover:bg-gray-50 transition-colors"
              >
                <div className="min-w-0">
                  <p className="font-medium text-gray-900 truncate">{p.title || p.topic || 'Untitled'}</p>
                  <p className="text-xs text-gray-400 mt-0.5">
                    Created {formatDate(p.created_at || p.createdAt)}
                    {p.topic && p.title !== p.topic && (
                      <span className="ml-2 text-gray-500">· {p.topic.slice(0, 60)}{p.topic.length > 60 ? '…' : ''}</span>
                    )}
                  </p>
                </div>
                <div className="flex items-center gap-3 ml-4 shrink-0">
                  <StatusBadge status={p.status} />
                  <button
                    onClick={() => navigate(`/projects/${p.id || p._id}`)}
                    className="flex items-center gap-1.5 text-sm font-medium text-primary-600 hover:text-primary-800 bg-primary-50 hover:bg-primary-100 px-3 py-1.5 rounded-lg transition-colors"
                  >
                    <FolderOpen size={14} />
                    Open
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
