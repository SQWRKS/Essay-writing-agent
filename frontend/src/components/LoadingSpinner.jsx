import React from 'react'

export default function LoadingSpinner({ size = 'md', className = '' }) {
  const sizeMap = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-2',
    lg: 'w-12 h-12 border-4',
  }
  const cls = sizeMap[size] || sizeMap.md

  return (
    <div
      className={`${cls} rounded-full border-gray-200 border-t-primary-600 animate-spin ${className}`}
      role="status"
      aria-label="Loading"
    />
  )
}
