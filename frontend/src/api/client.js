import axios from 'axios'

// VITE_API_URL can be set to an explicit backend URL (e.g. http://localhost:8000)
// when running without a proxy.  When left empty (the default), all requests use
// relative paths and are handled by the Vite dev-server proxy (dev) or the nginx
// reverse-proxy (Docker/production), which avoids CORS issues entirely.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? '',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred'
    return Promise.reject(new Error(message))
  }
)

export const getProjects = () => api.get('/projects')

export const createProject = (title, topic) =>
  api.post('/projects', { title, topic })

export const getProject = (id) => api.get(`/projects/${id}`)

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

export const getHealth = () => api.get('/api/health')

export const getConfig = () => api.get('/api/config')

export const updateConfig = (config) => api.post('/api/config', config)

export const updateProjectContent = (projectId, content) =>
  api.put(`/projects/${projectId}/content`, content)

export default api
