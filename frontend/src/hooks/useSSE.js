import { useState, useEffect, useRef, useCallback } from 'react'
import api from '../api/client'

// Derive the SSE base URL from the Axios instance base URL so that all
// backend connection settings remain in a single place (api/client.js).
const SSE_BASE_URL = api.defaults.baseURL || ''

export function useSSE(projectId) {
  const [events, setEvents] = useState([])
  const [connected, setConnected] = useState(false)
  const esRef = useRef(null)
  const reconnectTimer = useRef(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!projectId || !mountedRef.current) return
    if (esRef.current) {
      esRef.current.close()
    }

    const url = `${SSE_BASE_URL}/projects/${projectId}/events`
    const es = new EventSource(url)
    esRef.current = es

    es.onopen = () => {
      if (mountedRef.current) setConnected(true)
    }

    es.onmessage = (e) => {
      if (!mountedRef.current) return
      try {
        const data = JSON.parse(e.data)
        setEvents((prev) => [
          { ...data, _ts: Date.now() },
          ...prev.slice(0, 199),
        ])
      } catch {
        setEvents((prev) => [
          { raw: e.data, _ts: Date.now() },
          ...prev.slice(0, 199),
        ])
      }
    }

    es.onerror = () => {
      if (!mountedRef.current) return
      setConnected(false)
      es.close()
      esRef.current = null
      reconnectTimer.current = setTimeout(() => {
        if (mountedRef.current) connect()
      }, 3000)
    }
  }, [projectId])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      if (esRef.current) esRef.current.close()
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }
  }, [connect])

  return { events, connected }
}
