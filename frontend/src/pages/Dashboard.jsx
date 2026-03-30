import React, { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { PlusCircle, FolderOpen, AlertCircle, RefreshCw, BookOpen, Pause, Trash2, ChevronDown, ChevronUp, SlidersHorizontal } from 'lucide-react'
import { useProjects } from '../hooks/useProjects'
import { useConfig } from '../hooks/useConfig'
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
  const {
    projects,
    loading,
    error,
    refetch,
    createProject,
    uploadContextFile,
    pauseProject,
    deleteProject,
  } = useProjects()
  const { config, updateConfig, loading: configLoading } = useConfig()
  const navigate = useNavigate()

  const [topic, setTopic] = useState('')
  const [creating, setCreating] = useState(false)
  const [switchingMode, setSwitchingMode] = useState(false)
  const [createError, setCreateError] = useState(null)
  const [modeMsg, setModeMsg] = useState(null)
  const [projectActionMsg, setProjectActionMsg] = useState(null)
  const [busyProjectId, setBusyProjectId] = useState(null)

  // Fine-tune section state — all completely optional
  const [showFineTune, setShowFineTune] = useState(false)
  const [wordCountTarget, setWordCountTarget] = useState('')
  const [writingStyle, setWritingStyle] = useState('')
  const [contextText, setContextText] = useState('')
  const [contextFile, setContextFile] = useState(null)
  const [rubric, setRubric] = useState('')
  const fileInputRef = useRef(null)

  const activeQualityMode = (config?.QUALITY_MODE || 'quality').toLowerCase()

  const handleQualityModeChange = async (mode) => {
    setSwitchingMode(true)
    setModeMsg(null)
    try {
      await updateConfig({ QUALITY_MODE: mode, LLM_PROVIDER: 'anthropic' })
      setModeMsg({ type: 'success', text: `Quality mode switched to ${mode}.` })
    } catch (err) {
      setModeMsg({ type: 'error', text: err.message || 'Failed to switch quality mode.' })
    } finally {
      setSwitchingMode(false)
    }
  }

  const handleCreate = async (e) => {
    e.preventDefault()
    if (!topic.trim()) return
    setCreating(true)
    setCreateError(null)
    try {
      const title = topic.trim().slice(0, 80)

      // Build optional fine-tune settings — only include non-empty values so
      // that the backend behaves identically to before when nothing is set.
      const settings = {}
      const parsedWC = parseInt(wordCountTarget, 10)
      if (!isNaN(parsedWC) && parsedWC >= 100) settings.word_count_target = parsedWC
      if (writingStyle) settings.writing_style = writingStyle
      if (contextText.trim()) settings.context_text = contextText.trim()
      if (rubric.trim()) settings.rubric = rubric.trim()

      const project = await createProject(
        title,
        topic.trim(),
        Object.keys(settings).length > 0 ? settings : undefined,
      )

      // Upload context file if one was selected (non-blocking — errors are surfaced but don't abort)
      if (contextFile) {
        try {
          await uploadContextFile(project.id || project._id, contextFile)
        } catch (fileErr) {
          setCreateError(`Project created but file upload failed: ${fileErr.message}`)
        }
      }

      // Reset form
      setTopic('')
      setWordCountTarget('')
      setWritingStyle('')
      setContextText('')
      setContextFile(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
      setRubric('')
      setShowFineTune(false)

      navigate(`/projects/${project.id || project._id}`)
    } catch (err) {
      setCreateError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const handlePauseProject = async (projectId) => {
    setBusyProjectId(projectId)
    setProjectActionMsg(null)
    try {
      await pauseProject(projectId)
      setProjectActionMsg({ type: 'success', text: 'Project paused.' })
    } catch (err) {
      setProjectActionMsg({ type: 'error', text: err.message || 'Could not pause project.' })
    } finally {
      setBusyProjectId(null)
    }
  }

  const handleDeleteProject = async (projectId, projectTitle) => {
    const confirmed = window.confirm(`Delete "${projectTitle || 'this project'}"? This cannot be undone.`)
    if (!confirmed) return

    setBusyProjectId(projectId)
    setProjectActionMsg(null)
    try {
      await deleteProject(projectId)
      setProjectActionMsg({ type: 'success', text: 'Project deleted.' })
    } catch (err) {
      setProjectActionMsg({ type: 'error', text: err.message || 'Could not delete project.' })
    } finally {
      setBusyProjectId(null)
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
        <div className="flex items-start justify-between gap-4 flex-wrap mb-5">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Model Quality Mode</h2>
            <p className="text-sm text-gray-500 mt-1">
              {activeQualityMode === 'quality'
                ? 'Quality mode uses Claude Opus 4.6 for best writing quality.'
                : 'Balanced mode uses Claude Sonnet 4.6 for faster and lower-cost runs.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => handleQualityModeChange('quality')}
              disabled={configLoading || switchingMode || activeQualityMode === 'quality'}
              className="px-3 py-1.5 text-sm rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-50"
            >
              Quality
            </button>
            <button
              type="button"
              onClick={() => handleQualityModeChange('balanced')}
              disabled={configLoading || switchingMode || activeQualityMode === 'balanced'}
              className="px-3 py-1.5 text-sm rounded-lg border border-gray-200 hover:bg-gray-50 disabled:opacity-50"
            >
              Balanced
            </button>
          </div>
        </div>
        {modeMsg && (
          <div className={`mb-4 text-sm rounded-lg px-3 py-2 ${modeMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
            {modeMsg.text}
          </div>
        )}

        <h2 className="text-lg font-semibold text-gray-900 mb-4">Create New Project</h2>
        <form onSubmit={handleCreate} className="space-y-3">
          {/* Topic — required */}
          <div className="flex gap-3">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="Enter your essay topic…"
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              disabled={creating}
            />
          </div>

          {/* Fine-tune toggle */}
          <div>
            <button
              type="button"
              onClick={() => setShowFineTune((v) => !v)}
              className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-primary-600 transition-colors"
            >
              <SlidersHorizontal size={14} />
              Fine-tune options
              {showFineTune ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              <span className="ml-1 text-xs text-gray-400">(all optional)</span>
            </button>
          </div>

          {/* Fine-tune panel — collapsed by default */}
          {showFineTune && (
            <div className="border border-gray-200 rounded-xl p-4 space-y-4 bg-gray-50">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Word count target */}
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    Target word count
                  </label>
                  <input
                    type="number"
                    min={100}
                    max={50000}
                    step={100}
                    value={wordCountTarget}
                    onChange={(e) => setWordCountTarget(e.target.value)}
                    placeholder="Default (varies by section)"
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    disabled={creating}
                  />
                  <p className="mt-0.5 text-xs text-gray-400">
                    Section targets are scaled proportionally.
                  </p>
                </div>

                {/* Writing style */}
                <div>
                  <label className="block text-xs font-medium text-gray-600 mb-1">
                    Writing style
                  </label>
                  <select
                    value={writingStyle}
                    onChange={(e) => setWritingStyle(e.target.value)}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    disabled={creating}
                  >
                    <option value="">Default</option>
                    <option value="academic">Academic</option>
                    <option value="argumentative">Argumentative</option>
                    <option value="persuasive">Persuasive</option>
                    <option value="analytical">Analytical</option>
                    <option value="critical">Critical</option>
                    <option value="expository">Expository</option>
                  </select>
                </div>
              </div>

              {/* Context text */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Additional context <span className="font-normal text-gray-400">(background knowledge, domain notes)</span>
                </label>
                <textarea
                  value={contextText}
                  onChange={(e) => setContextText(e.target.value)}
                  placeholder="Paste any background context, key terms, or domain-specific information the agents should be aware of…"
                  rows={3}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-y"
                  disabled={creating}
                />
              </div>

              {/* Context file upload */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Context file <span className="font-normal text-gray-400">(.txt, .docx, or .pdf)</span>
                </label>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.docx,.pdf"
                  onChange={(e) => setContextFile(e.target.files?.[0] || null)}
                  className="block w-full text-sm text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                  disabled={creating}
                />
                {contextFile && (
                  <p className="mt-1 text-xs text-green-600">
                    Selected: {contextFile.name} ({(contextFile.size / 1024).toFixed(1)} KB)
                  </p>
                )}
              </div>

              {/* Marking rubric */}
              <div>
                <label className="block text-xs font-medium text-gray-600 mb-1">
                  Marking rubric <span className="font-normal text-gray-400">(grading criteria for the reviewer agent)</span>
                </label>
                <textarea
                  value={rubric}
                  onChange={(e) => setRubric(e.target.value)}
                  placeholder="e.g. 30% critical analysis, 25% evidence quality, 25% structure and flow, 20% originality…"
                  rows={3}
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-y"
                  disabled={creating}
                />
              </div>
            </div>
          )}

          {/* Submit */}
          <div className="flex justify-end">
            <button
              type="submit"
              disabled={creating || !topic.trim()}
              className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {creating ? <LoadingSpinner size="sm" /> : <PlusCircle size={16} />}
              {creating ? 'Creating…' : 'Create Project'}
            </button>
          </div>
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

        {projectActionMsg && (
          <div className={`mx-6 mt-4 rounded-lg px-3 py-2 text-sm ${projectActionMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
            {projectActionMsg.text}
          </div>
        )}

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
                    onClick={() => handlePauseProject(p.id || p._id)}
                    disabled={busyProjectId === (p.id || p._id) || ['paused', 'completed', 'failed'].includes((p.status || '').toLowerCase())}
                    className="flex items-center gap-1.5 text-sm font-medium text-amber-700 hover:text-amber-800 bg-amber-50 hover:bg-amber-100 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
                  >
                    <Pause size={14} />
                    Pause
                  </button>
                  <button
                    onClick={() => handleDeleteProject(p.id || p._id, p.title || p.topic)}
                    disabled={busyProjectId === (p.id || p._id) || (p.status || '').toLowerCase() === 'running'}
                    className="flex items-center gap-1.5 text-sm font-medium text-red-700 hover:text-red-800 bg-red-50 hover:bg-red-100 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-40"
                  >
                    <Trash2 size={14} />
                    Delete
                  </button>
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
