import React, { useState } from 'react'
import { useParams } from 'react-router-dom'
import { Download, FileText, File, BookOpen, AlertCircle, CheckCircle2, Clock } from 'lucide-react'
import { exportProject } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'

const FORMATS = [
  {
    id: 'txt',
    label: 'Plain Text',
    ext: '.txt',
    mime: 'text/plain',
    icon: FileText,
    description: 'Simple text file, compatible with any editor.',
    color: 'border-gray-200 hover:border-gray-400',
    iconColor: 'text-gray-600 bg-gray-100',
  },
  {
    id: 'pdf',
    label: 'PDF Document',
    ext: '.pdf',
    mime: 'application/pdf',
    icon: BookOpen,
    description: 'Formatted PDF ready for submission.',
    color: 'border-red-200 hover:border-red-400',
    iconColor: 'text-red-600 bg-red-50',
  },
  {
    id: 'docx',
    label: 'Word Document',
    ext: '.docx',
    mime: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    icon: File,
    description: 'Microsoft Word format, fully editable.',
    color: 'border-blue-200 hover:border-blue-400',
    iconColor: 'text-blue-600 bg-blue-50',
  },
]

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString()
}

export default function ExportPanel() {
  const { id } = useParams()
  const [loading, setLoading] = useState({})
  const [history, setHistory] = useState([])
  const [error, setError] = useState(null)

  const handleExport = async (format) => {
    setLoading((prev) => ({ ...prev, [format.id]: true }))
    setError(null)

    const startTs = Date.now()
    try {
      const res = await exportProject(id, format.id)
      const blob = new Blob([res.data], { type: format.mime })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `essay-project-${id}${format.ext}`
      document.body.appendChild(a)
      a.click()
      setTimeout(() => {
        URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }, 200)

      setHistory((prev) => [
        {
          id: Date.now(),
          format: format.id.toUpperCase(),
          label: format.label,
          ts: Date.now(),
          duration: Date.now() - startTs,
          status: 'success',
        },
        ...prev.slice(0, 19),
      ])
    } catch (err) {
      setError(err.message)
      setHistory((prev) => [
        {
          id: Date.now(),
          format: format.id.toUpperCase(),
          label: format.label,
          ts: Date.now(),
          status: 'failed',
          error: err.message,
        },
        ...prev.slice(0, 19),
      ])
    } finally {
      setLoading((prev) => ({ ...prev, [format.id]: false }))
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Export Document</h1>
        <p className="text-gray-500 text-sm mt-1">
          Download your completed essay in your preferred format.
        </p>
      </div>

      {error && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-xl p-4 text-red-700 text-sm">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {/* Export Cards */}
      <div className="grid gap-4">
        {FORMATS.map((fmt) => {
          const Icon = fmt.icon
          const isLoading = loading[fmt.id]
          return (
            <div
              key={fmt.id}
              className={`bg-white rounded-xl border-2 ${fmt.color} shadow-sm p-5 flex items-center justify-between transition-colors`}
            >
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-xl ${fmt.iconColor}`}>
                  <Icon size={22} />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{fmt.label}</h3>
                  <p className="text-sm text-gray-500">{fmt.description}</p>
                </div>
              </div>
              <button
                onClick={() => handleExport(fmt)}
                disabled={isLoading}
                className="flex items-center gap-2 bg-gray-900 hover:bg-gray-700 text-white px-4 py-2.5 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 whitespace-nowrap"
              >
                {isLoading ? (
                  <><LoadingSpinner size="sm" /> Exporting…</>
                ) : (
                  <><Download size={14} /> Export as {fmt.id.toUpperCase()}</>
                )}
              </button>
            </div>
          )
        })}
      </div>

      {/* Export History */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
        <div className="px-6 py-4 border-b border-gray-100">
          <h2 className="font-semibold text-gray-900 flex items-center gap-2">
            <Clock size={16} /> Export History
          </h2>
        </div>
        {history.length === 0 ? (
          <div className="py-12 text-center text-sm text-gray-400">
            No exports yet. Use the buttons above to download your essay.
          </div>
        ) : (
          <ul className="divide-y divide-gray-100">
            {history.map((h) => (
              <li key={h.id} className="flex items-center justify-between px-6 py-3">
                <div className="flex items-center gap-3">
                  {h.status === 'success' ? (
                    <CheckCircle2 size={16} className="text-green-500" />
                  ) : (
                    <AlertCircle size={16} className="text-red-500" />
                  )}
                  <div>
                    <span className="text-sm font-medium text-gray-800">{h.label}</span>
                    {h.duration && (
                      <span className="ml-2 text-xs text-gray-400">({h.duration}ms)</span>
                    )}
                    {h.error && (
                      <p className="text-xs text-red-500 mt-0.5">{h.error}</p>
                    )}
                  </div>
                </div>
                <span className="text-xs text-gray-400">{formatTime(h.ts)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
