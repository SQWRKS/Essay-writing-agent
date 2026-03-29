import React, { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { Save, RefreshCw, BookOpen, Quote, AlertCircle, RotateCcw } from 'lucide-react'
import { getProject, runAgent, updateProjectContent } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'

const DEFAULT_SECTIONS = [
  'Introduction',
  'Literature Review',
  'Methodology',
  'Results',
  'Discussion',
  'Conclusion',
  'References',
]

function wordCount(text) {
  if (!text) return 0
  return text.trim().split(/\s+/).filter(Boolean).length
}

function parseContent(project) {
  if (!project) return {}
  if (typeof project.content === 'object' && project.content !== null) {
    return project.content
  }
  if (typeof project.content === 'string') {
    try { return JSON.parse(project.content) } catch { /* ignore */ }
    return { Introduction: project.content }
  }
  return {}
}

function parseCitations(project) {
  if (!project?.citations) return []
  if (Array.isArray(project.citations)) return project.citations
  if (typeof project.citations === 'object') return Object.values(project.citations)
  return []
}

export default function DocumentEditor() {
  const { id } = useParams()
  const [project, setProject] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [sections, setSections] = useState({})
  const [activeSection, setActiveSection] = useState(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState(null)
  const [rerunning, setRerunning] = useState(false)

  const fetchProject = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getProject(id)
      setProject(res.data)
      const parsed = parseContent(res.data)
      setSections(parsed)
      if (!activeSection) {
        const first = Object.keys(parsed)[0] || DEFAULT_SECTIONS[0]
        setActiveSection(first)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [id, activeSection])

  useEffect(() => { fetchProject() }, [id])

  const allSections = [...new Set([
    ...DEFAULT_SECTIONS,
    ...Object.keys(sections),
  ])]

  const handleChange = (value) => {
    setSections((prev) => ({ ...prev, [activeSection]: value }))
    setDirty(true)
  }

  const handleSave = async () => {
    setSaving(true)
    setSaveMsg(null)
    try {
      await updateProjectContent(id, sections)
      setDirty(false)
      setSaveMsg({ type: 'success', text: 'Saved.' })
    } catch (err) {
      setSaveMsg({ type: 'error', text: err.message })
    } finally {
      setSaving(false)
      setTimeout(() => setSaveMsg(null), 3000)
    }
  }

  const handleRerunWriter = async () => {
    setRerunning(true)
    setSaveMsg(null)
    try {
      await runAgent(id, 'WriterAgent', { section: activeSection })
      setSaveMsg({ type: 'success', text: `Writer agent re-queued for "${activeSection}".` })
      setTimeout(fetchProject, 3000)
    } catch (err) {
      setSaveMsg({ type: 'error', text: err.message })
    } finally {
      setRerunning(false)
    }
  }

  // No localStorage fallback needed — content is persisted on the backend via handleSave
  

  const citations = parseCitations(project)
  const activeContent = sections[activeSection] || ''

  if (loading) return <div className="flex justify-center py-20"><LoadingSpinner size="lg" /></div>
  if (error) return (
    <div className="max-w-xl mx-auto mt-12 bg-red-50 border border-red-200 rounded-xl p-6 text-center">
      <AlertCircle className="mx-auto mb-2 text-red-500" size={32} />
      <p className="text-red-700">{error}</p>
      <button onClick={fetchProject} className="mt-3 text-sm text-primary-600 underline">Retry</button>
    </div>
  )

  return (
    <div className="max-w-7xl mx-auto">
      {/* Page header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Document Editor</h1>
          <p className="text-gray-500 text-sm">{project?.title || project?.topic}</p>
        </div>
        <div className="flex items-center gap-2">
          {dirty && (
            <span className="text-xs text-yellow-600 bg-yellow-50 px-2 py-1 rounded">Unsaved changes</span>
          )}
          <button
            onClick={fetchProject}
            className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-50"
          >
            <RefreshCw size={14} />
            Reload
          </button>
          <button
            onClick={handleSave}
            disabled={saving || !dirty}
            className="flex items-center gap-2 bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50"
          >
            {saving ? <LoadingSpinner size="sm" /> : <Save size={14} />}
            Save
          </button>
        </div>
      </div>

      {saveMsg && (
        <div className={`mb-4 flex items-center gap-2 text-sm p-3 rounded-lg ${
          saveMsg.type === 'success' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
        }`}>
          {saveMsg.text}
        </div>
      )}

      <div className="flex gap-4 h-[calc(100vh-220px)]">
        {/* Section Sidebar */}
        <div className="w-52 shrink-0 bg-white rounded-xl border border-gray-200 shadow-sm overflow-y-auto scrollbar-thin">
          <div className="px-4 py-3 border-b border-gray-100">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <BookOpen size={12} /> Sections
            </p>
          </div>
          <ul className="p-2 space-y-0.5">
            {allSections.map((sec) => {
              const wc = wordCount(sections[sec])
              return (
                <li key={sec}>
                  <button
                    onClick={() => setActiveSection(sec)}
                    className={`w-full text-left px-3 py-2.5 rounded-lg text-sm transition-colors ${
                      activeSection === sec
                        ? 'bg-primary-600 text-white'
                        : 'text-gray-700 hover:bg-gray-100'
                    }`}
                  >
                    <span className="block truncate">{sec}</span>
                    <span className={`text-xs ${activeSection === sec ? 'text-primary-200' : 'text-gray-400'}`}>
                      {wc} words
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
        </div>

        {/* Editor */}
        <div className="flex-1 flex flex-col bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
            <h2 className="font-semibold text-gray-900">{activeSection}</h2>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-400">{wordCount(activeContent)} words</span>
              <button
                onClick={handleRerunWriter}
                disabled={rerunning}
                className="flex items-center gap-1.5 text-sm text-primary-600 hover:text-primary-800 border border-primary-200 hover:border-primary-400 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-50"
              >
                {rerunning ? <LoadingSpinner size="sm" /> : <RotateCcw size={13} />}
                Re-run Writer
              </button>
            </div>
          </div>
          <textarea
            value={activeContent}
            onChange={(e) => handleChange(e.target.value)}
            placeholder={`Start writing the ${activeSection} section…`}
            className="flex-1 p-5 text-sm leading-relaxed text-gray-800 resize-none focus:outline-none font-serif"
          />
        </div>

        {/* Citations Panel */}
        <div className="w-64 shrink-0 bg-white rounded-xl border border-gray-200 shadow-sm overflow-y-auto scrollbar-thin">
          <div className="px-4 py-3 border-b border-gray-100">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <Quote size={12} /> Citations
            </p>
          </div>
          {citations.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-gray-400">
              No citations found yet. Run the Citation Agent to generate references.
            </div>
          ) : (
            <ul className="p-3 space-y-2">
              {citations.map((cit, i) => (
                <li key={i} className="text-xs bg-gray-50 rounded-lg p-3 text-gray-700 leading-relaxed border border-gray-100">
                  {typeof cit === 'string'
                    ? cit
                    : (cit.text || cit.citation || cit.reference || JSON.stringify(cit))}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
