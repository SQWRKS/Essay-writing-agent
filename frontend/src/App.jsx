import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ProjectView from './pages/ProjectView'
import AgentMonitor from './pages/AgentMonitor'
import DocumentEditor from './pages/DocumentEditor'
import ExportPanel from './pages/ExportPanel'
import ApiMonitor from './pages/ApiMonitor'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/projects/:id" element={<ProjectView />} />
          <Route path="/projects/:id/agents" element={<AgentMonitor />} />
          <Route path="/projects/:id/editor" element={<DocumentEditor />} />
          <Route path="/projects/:id/export" element={<ExportPanel />} />
          <Route path="/api-monitor" element={<ApiMonitor />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
