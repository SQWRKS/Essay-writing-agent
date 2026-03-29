import { useState, useEffect, useCallback } from 'react'
import { getProjects, createProject as apiCreateProject } from '../api/client'

export function useProjects() {
  const [projects, setProjects] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchProjects = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await getProjects()
      setProjects(Array.isArray(res.data) ? res.data : (res.data?.projects ?? []))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    fetchProjects()
  }, [fetchProjects])

  const createProject = useCallback(async (title, topic) => {
    const res = await apiCreateProject(title, topic)
    const newProject = res.data
    setProjects((prev) => [newProject, ...prev])
    return newProject
  }, [])

  return { projects, loading, error, refetch: fetchProjects, createProject }
}
