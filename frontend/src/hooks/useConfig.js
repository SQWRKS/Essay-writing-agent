import { useState, useEffect, useCallback } from 'react'
import { getConfig, updateConfig as apiUpdateConfig } from '../api/client'

export function useConfig() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchConfig = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getConfig()
      setConfig(res.data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchConfig()
  }, [fetchConfig])

  const updateConfig = useCallback(async (newConfig) => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiUpdateConfig(newConfig)
      setConfig(res.data)
      return res.data
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [])

  return { config, loading, error, updateConfig, refetch: fetchConfig }
}
