import React, { useState, useEffect } from 'react'
import { getHealth } from '../api/client'

export default function ConnectionStatus() {
  const [healthy, setHealthy] = useState(null)

  useEffect(() => {
    const check = async () => {
      try {
        await getHealth()
        setHealthy(true)
      } catch {
        setHealthy(false)
      }
    }
    check()
    const timer = setInterval(check, 5000)
    return () => clearInterval(timer)
  }, [])

  if (healthy === null) {
    return (
      <span className="flex items-center gap-1.5 text-xs text-gray-400">
        <span className="w-2 h-2 rounded-full bg-gray-300 animate-pulse" />
        Checking…
      </span>
    )
  }

  return (
    <span className={`flex items-center gap-1.5 text-xs font-medium ${healthy ? 'text-green-600' : 'text-red-500'}`}>
      <span className={`w-2 h-2 rounded-full ${healthy ? 'bg-green-500' : 'bg-red-500 animate-pulse'}`} />
      {healthy ? 'API Online' : 'API Offline'}
    </span>
  )
}
