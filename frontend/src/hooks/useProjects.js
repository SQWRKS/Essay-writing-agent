import { useState, useEffect, useCallback } from 'react'
import {
  getProjects,
  createProject as apiCreateProject,
  uploadContextFile as apiUploadContextFile,
  pauseProject as apiPauseProject,
  deleteProject as apiDeleteProject,
} from '../api/client'

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

  const createProject = useCallback(async (title, topic, settings) => {
    const res = await apiCreateProject(title, topic, settings)
    const newProject = res.data
    setProjects((prev) => [newProject, ...prev])
    return newProject
  }, [])

  const uploadContextFile = useCallback(async (projectId, file) => {
    return apiUploadContextFile(projectId, file)
  }, [])

  const pauseProject = useCallback(async (projectId) => {
    await apiPauseProject(projectId)
    setProjects((prev) => prev.map((project) => {
      if ((project.id || project._id) !== projectId) return project
      return { ...project, status: 'paused' }
    }))
  }, [])

  const deleteProject = useCallback(async (projectId) => {
    await apiDeleteProject(projectId)
    setProjects((prev) => prev.filter((project) => (project.id || project._id) !== projectId))
  }, [])

  return {
    projects,
    loading,
    error,
    refetch: fetchProjects,
    createProject,
    uploadContextFile,
    pauseProject,
    deleteProject,
  }
}
