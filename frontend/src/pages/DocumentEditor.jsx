import React, { useState, useEffect, useCallback } from 'react'
import { useParams } from 'react-router-dom'
import { Save, RefreshCw, BookOpen, Quote, AlertCircle, RotateCcw, Image as ImageIcon, Globe } from 'lucide-react'
import api from '../api/client'
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
  if (text == null) return 0
  const normalized = typeof text === 'string' ? text : String(text)
  if (!normalized.trim()) return 0
  return normalized.trim().split(/\s+/).filter(Boolean).length
}

function normalizeSectionValue(value) {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    return value
      .map((item) => (typeof item === 'string' ? item : JSON.stringify(item)))
      .join('\n\n')
  }
  if (typeof value === 'object') {
    if (typeof value.text === 'string') return value.text
    if (typeof value.content === 'string') return value.content
    return JSON.stringify(value, null, 2)
  }
  return String(value)
}

function parseContent(project) {
  if (!project) return {}
  let rawContent = project.content

  if (typeof rawContent === 'string') {
    try {
      rawContent = JSON.parse(rawContent)
    } catch {
      return { Introduction: rawContent }
    }
  }

  if (!rawContent || typeof rawContent !== 'object') return {}

  const sectionSource =
    rawContent.sections && typeof rawContent.sections === 'object' && !Array.isArray(rawContent.sections)
      ? rawContent.sections
      : rawContent

  const normalized = {}
  Object.entries(sectionSource).forEach(([key, value]) => {
    normalized[key] = normalizeSectionValue(value)
  })

  return normalized
}

function parseContentEnvelope(project) {
  if (!project) return { sections: {}, metadata: {} }

  let rawContent = project.content
  if (typeof rawContent === 'string') {
    try {
      rawContent = JSON.parse(rawContent)
    } catch {
      return { sections: { Introduction: rawContent }, metadata: {} }
    }
  }

  if (!rawContent || typeof rawContent !== 'object') return { sections: {}, metadata: {} }

  const sections =
    rawContent.sections && typeof rawContent.sections === 'object' && !Array.isArray(rawContent.sections)
      ? rawContent.sections
      : rawContent

  const metadata = rawContent.metadata && typeof rawContent.metadata === 'object'
    ? rawContent.metadata
    : {}

  return { sections, metadata }
}

function toSectionKey(sectionName) {
  return String(sectionName || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
}

function parseCitations(project) {
  if (!project) return []
  let content = project.content
  if (typeof content === 'string') {
    try {
      content = JSON.parse(content)
    } catch {
      content = {}
    }
  }
  const raw = content?.metadata?.citations || project?.citations
  if (!raw) return []
  if (Array.isArray(raw)) return raw
  if (typeof raw === 'object') return Object.values(raw)
  return []
}

function parseFigures(project) {
  if (!project) return []
  let content = project.content
  if (typeof content === 'string') {
    try {
      content = JSON.parse(content)
    } catch {
      content = {}
    }
  }
  const raw = content?.metadata?.figures
  if (!Array.isArray(raw)) return []
  return raw
}

function parseSources(project) {
  if (!project) return []
  let content = project.content
  if (typeof content === 'string') {
    try {
      content = JSON.parse(content)
    } catch {
      content = {}
    }
  }
  const raw = content?.metadata?.sources
  return Array.isArray(raw) ? raw : []
}

function toFigureUrl(pathOrUrl) {
  if (!pathOrUrl) return ''
  if (pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')) return pathOrUrl

  const staticMarker = '/static/figures/'
  const markerIndex = pathOrUrl.indexOf(staticMarker)
  if (markerIndex >= 0) {
    return `${api.defaults.baseURL}${pathOrUrl.slice(markerIndex)}`
  }

  if (pathOrUrl.startsWith('/')) return `${api.defaults.baseURL}${pathOrUrl}`
  return ''
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
  const [metadata, setMetadata] = useState({})

  const fetchProject = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getProject(id)
      setProject(res.data)
      const parsed = parseContent(res.data)
      setSections(parsed)
      const envelope = parseContentEnvelope(res.data)
      setMetadata(envelope.metadata || {})
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

  useEffect(() => { fetchProject() }, [fetchProject])

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
      const serializedSections = {}
      Object.entries(sections).forEach(([key, value]) => {
        const normalizedKey = toSectionKey(key)
        serializedSections[normalizedKey || key] = normalizeSectionValue(value)
      })

      await updateProjectContent(id, {
        sections: serializedSections,
        metadata: {
          ...metadata,
          topic: metadata?.topic || project?.topic || '',
        },
      })
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
      await runAgent(id, 'writer', {
        section: toSectionKey(activeSection),
        topic: project?.topic || '',
        word_count: Math.max(400, wordCount(activeContent) || 600),
        research_data: {
          sources,
          research_summary: metadata?.research?.summary || '',
        },
      })
      setSaveMsg({ type: 'success', text: `Writer agent re-run for "${activeSection}".` })
      setTimeout(fetchProject, 3000)
    } catch (err) {
      setSaveMsg({ type: 'error', text: err.message })
    } finally {
      setRerunning(false)
    }
  }

  // No localStorage fallback needed — content is persisted on the backend via handleSave
  

  const citations = parseCitations(project)
  const figures = parseFigures(project)
  const sources = parseSources(project)
  const activeContent = normalizeSectionValue(sections[activeSection])

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

          <div className="px-4 py-3 border-y border-gray-100 mt-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <ImageIcon size={12} /> Figures
            </p>
          </div>
          {figures.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-gray-400">
              No figures generated yet.
            </div>
          ) : (
            <ul className="p-3 space-y-3">
              {figures.map((fig, i) => {
                const src = toFigureUrl(fig.url || fig.path)
                return (
                  <li key={`${fig.title || 'fig'}-${i}`} className="text-xs bg-gray-50 rounded-lg p-3 border border-gray-100">
                    {src && (
                      <img
                        src={src}
                        alt={fig.title || `Figure ${i + 1}`}
                        className="w-full h-auto rounded border border-gray-200 mb-2"
                      />
                    )}
                    <p className="font-semibold text-gray-700">{fig.title || `Figure ${i + 1}`}</p>
                    {fig.description && <p className="text-gray-500 mt-1">{fig.description}</p>}
                  </li>
                )
              })}
            </ul>
          )}

          <div className="px-4 py-3 border-y border-gray-100 mt-2">
            <p className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-1.5">
              <Globe size={12} /> Source Provenance
            </p>
          </div>
          {sources.length === 0 ? (
            <div className="px-4 py-6 text-center text-xs text-gray-400">
              No source metadata available yet.
            </div>
          ) : (
            <ul className="p-3 space-y-2">
              {sources.slice(0, 12).map((src, i) => (
                <li key={`${src.doi || src.url || src.title || 'source'}-${i}`} className="text-xs bg-gray-50 rounded-lg p-3 border border-gray-100">
                  <p className="font-semibold text-gray-700 leading-snug">{src.title || 'Untitled source'}</p>
                  <p className="text-gray-500 mt-1">
                    {(src.year || 'n.d.')} · {(src.source || 'unknown').replace(/_/g, ' ')}
                    {typeof src.relevance_score === 'number' && ` · score ${src.relevance_score}`}
                  </p>
                  {src.url && (
                    <a
                      href={src.url}
                      target="_blank"
                      rel="noreferrer"
                      className="text-primary-600 hover:text-primary-800 mt-1 inline-block break-all"
                    >
                      {src.url}
                    </a>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
