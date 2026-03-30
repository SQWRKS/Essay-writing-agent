import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  // Some endpoints can be delayed while long-running generation writes to DB.
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out while waiting for the API. The pipeline may still be running.'))
    }

    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

export const getProjects = () => api.get('/projects')

export const createProject = (title, topic, settings) =>
  api.post('/projects', {
    title,
    topic,
    ...(settings && Object.keys(settings).length > 0 ? { settings } : {}),
  })

export const uploadContextFile = (projectId, file) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post(`/projects/${projectId}/upload-context`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  })
}

export const getProject = (id) => api.get(`/projects/${id}`)

export const pauseProject = (projectId) =>
  api.post(`/projects/${projectId}/pause`)

export const deleteProject = (projectId) =>
  api.delete(`/projects/${projectId}`)

export const runAgent = (projectId, agentName, inputData) =>
  api.post(`/projects/${projectId}/run-agent`, { agent_name: agentName, input_data: inputData })

export const runPipeline = (projectId) =>
  api.post(`/projects/${projectId}/run`)

export const getTasks = (projectId) =>
  api.get(`/projects/${projectId}/tasks`)

export const exportProject = (projectId, format) =>
  api.get(`/projects/${projectId}/export`, {
    params: { format },
    responseType: 'blob',
  })

export const getLogs = (page = 1, limit = 50) =>
  api.get('/api/logs', { params: { page, limit } })

export const getHealth = () => api.get('/api/health', { timeout: 5000 })

export const getConfig = () => api.get('/api/config')

export const updateConfig = (config) => api.post('/api/config', config)

export const updateProjectContent = (projectId, content) =>
  api.put(`/projects/${projectId}/content`, content)

export default api
