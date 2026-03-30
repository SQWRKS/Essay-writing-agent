import React from 'react'

const STATUS_STYLES = {
  pending:   'bg-gray-100 text-gray-600',
  queued:    'bg-gray-100 text-gray-600',
  running:   'bg-blue-100 text-blue-700',
  paused:    'bg-amber-100 text-amber-800',
  completed: 'bg-green-100 text-green-700',
  success:   'bg-green-100 text-green-700',
  failed:    'bg-red-100 text-red-700',
  error:     'bg-red-100 text-red-700',
  cancelled: 'bg-yellow-100 text-yellow-700',
}

const STATUS_DOT = {
  pending:   'bg-gray-400',
  queued:    'bg-gray-400',
  running:   'bg-blue-500 animate-pulse',
  paused:    'bg-amber-500',
  completed: 'bg-green-500',
  success:   'bg-green-500',
  failed:    'bg-red-500',
  error:     'bg-red-500',
  cancelled: 'bg-yellow-500',
}

export default function StatusBadge({ status = 'pending', className = '' }) {
  const normalized = (status || 'pending').toLowerCase()
  const style = STATUS_STYLES[normalized] || 'bg-gray-100 text-gray-600'
  const dot = STATUS_DOT[normalized] || 'bg-gray-400'

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium ${style} ${className}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${dot}`} />
      {normalized.charAt(0).toUpperCase() + normalized.slice(1)}
    </span>
  )
}
