import React from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import ConnectionStatus from './ConnectionStatus'

export default function Layout() {
  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      <Sidebar />
      <div className="flex flex-col flex-1 overflow-hidden">
        <header className="flex items-center justify-between px-6 py-3 bg-white border-b border-gray-200 shadow-sm">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-gray-600">System Status</span>
            <ConnectionStatus />
          </div>
          <span className="text-xs text-gray-400">Essay Writing Agent v0.1.0</span>
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
