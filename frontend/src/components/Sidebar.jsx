import React from 'react'
import { NavLink, useParams } from 'react-router-dom'
import {
  LayoutDashboard,
  FolderOpen,
  Bot,
  FileText,
  Download,
  Activity,
  PenTool,
} from 'lucide-react'

const navLinkClass = ({ isActive }) =>
  `flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
    isActive
      ? 'bg-primary-600 text-white shadow-sm'
      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
  }`

export default function Sidebar() {
  const { id: projectId } = useParams()

  return (
    <aside className="w-60 bg-white border-r border-gray-200 flex flex-col shadow-sm">
      <div className="flex items-center gap-3 px-5 py-5 border-b border-gray-100">
        <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center">
          <PenTool size={16} className="text-white" />
        </div>
        <span className="font-bold text-gray-900 text-lg">Essay Agent</span>
      </div>

      <nav className="flex-1 p-3 space-y-1 overflow-y-auto scrollbar-thin">
        <p className="px-3 pt-2 pb-1 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Main
        </p>
        <NavLink to="/" end className={navLinkClass}>
          <LayoutDashboard size={16} />
          Dashboard
        </NavLink>
        <NavLink to="/api-monitor" className={navLinkClass}>
          <Activity size={16} />
          API Monitor
        </NavLink>

        {projectId && (
          <>
            <p className="px-3 pt-4 pb-1 text-xs font-semibold text-gray-400 uppercase tracking-wider">
              Project
            </p>
            <NavLink to={`/projects/${projectId}`} end className={navLinkClass}>
              <FolderOpen size={16} />
              Overview
            </NavLink>
            <NavLink to={`/projects/${projectId}/agents`} className={navLinkClass}>
              <Bot size={16} />
              Agents
            </NavLink>
            <NavLink to={`/projects/${projectId}/editor`} className={navLinkClass}>
              <FileText size={16} />
              Editor
            </NavLink>
            <NavLink to={`/projects/${projectId}/export`} className={navLinkClass}>
              <Download size={16} />
              Export
            </NavLink>
          </>
        )}
      </nav>

      <div className="p-4 border-t border-gray-100">
        <p className="text-xs text-gray-400 text-center">AI Academic Writing System</p>
      </div>
    </aside>
  )
}
