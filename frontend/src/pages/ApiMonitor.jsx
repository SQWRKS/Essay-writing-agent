import React, { useState, useEffect, useCallback } from 'react'
import {
  Activity, RefreshCw, ChevronLeft, ChevronRight, AlertCircle,
  Clock, TrendingUp, XCircle,
} from 'lucide-react'
import { getLogs } from '../api/client'
import LoadingSpinner from '../components/LoadingSpinner'

function statusColor(status) {
  if (!status) return 'text-gray-500'
  const s = Number(status)
  if (s >= 500) return 'text-red-600 bg-red-50'
  if (s >= 400) return 'text-yellow-600 bg-yellow-50'
  if (s >= 200) return 'text-green-600 bg-green-50'
  return 'text-gray-600 bg-gray-50'
}

function formatDate(d) {
  if (!d) return '—'
  try { return new Date(d).toLocaleString() } catch { return d }
}

function SummaryCard({ icon: Icon, label, value, color = 'text-gray-800' }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-5 flex items-center gap-4">
      <div className={`p-3 rounded-lg bg-gray-100 ${color}`}>
        <Icon size={20} />
      </div>
      <div>
        <p className="text-xs font-medium text-gray-500">{label}</p>
        <p className={`text-2xl font-bold ${color}`}>{value}</p>
      </div>
    </div>
  )
}

export default function ApiMonitor() {
  const [logs, setLogs] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(false)
  const LIMIT = 50

  const fetchLogs = useCallback(async (p = 1) => {
    setLoading(true)
    setError(null)
    try {
      const res = await getLogs(p, LIMIT)
      const raw = res.data
      const items = Array.isArray(raw) ? raw : (raw?.logs ?? raw?.items ?? [])
      setLogs(items)
      setHasMore(items.length === LIMIT)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchLogs(page) }, [fetchLogs, page])

  // Auto-refresh every 5 seconds
  useEffect(() => {
    const timer = setInterval(() => fetchLogs(page), 5000)
    return () => clearInterval(timer)
  }, [fetchLogs, page])

  const totalCalls = logs.length
  const avgDuration = totalCalls
    ? (logs.reduce((s, l) => s + (l.duration_ms || l.duration || 0), 0) / totalCalls).toFixed(0)
    : 0
  const errorCount = logs.filter((l) => {
    const s = l.status_code || l.status || 0
    return Number(s) >= 400
  }).length
  const errorRate = totalCalls ? ((errorCount / totalCalls) * 100).toFixed(1) : 0

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Activity size={22} /> API Monitor
          </h1>
          <p className="text-gray-500 text-sm mt-0.5">Live API call logs · auto-refreshes every 5s</p>
        </div>
        <button
          onClick={() => fetchLogs(page)}
          disabled={loading}
          className="flex items-center gap-1.5 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-200 rounded-lg hover:bg-gray-50"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <SummaryCard
          icon={Activity}
          label="Total API Calls"
          value={loading ? '…' : totalCalls}
          color="text-primary-600"
        />
        <SummaryCard
          icon={Clock}
          label="Avg Duration (ms)"
          value={loading ? '…' : avgDuration}
          color="text-blue-600"
        />
        <SummaryCard
          icon={XCircle}
          label="Error Rate"
          value={loading ? '…' : `${errorRate}%`}
          color={Number(errorRate) > 10 ? 'text-red-600' : 'text-green-600'}
        />
      </div>

      {/* Logs Table */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <h2 className="font-semibold text-gray-900">Recent API Logs</h2>
          {loading && <LoadingSpinner size="sm" />}
        </div>

        {error ? (
          <div className="py-12 text-center">
            <AlertCircle className="mx-auto mb-2 text-red-400" size={28} />
            <p className="text-red-600 text-sm">{error}</p>
            <button onClick={() => fetchLogs(page)} className="mt-2 text-sm text-primary-600 underline">
              Retry
            </button>
          </div>
        ) : logs.length === 0 && !loading ? (
          <div className="py-16 text-center text-gray-400 text-sm">
            No API logs found. Logs will appear here as agents make API calls.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                  <th className="px-5 py-3">Timestamp</th>
                  <th className="px-5 py-3">Method</th>
                  <th className="px-5 py-3">Endpoint</th>
                  <th className="px-5 py-3">Agent</th>
                  <th className="px-5 py-3">Duration</th>
                  <th className="px-5 py-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {logs.map((log, i) => {
                  const sc = log.status_code || log.status
                  const colorCls = statusColor(sc)
                  return (
                    <tr key={log.id || i} className="hover:bg-gray-50 transition-colors">
                      <td className="px-5 py-3 text-gray-500 whitespace-nowrap text-xs">
                        {formatDate(log.timestamp || log.created_at)}
                      </td>
                      <td className="px-5 py-3">
                        <span className="font-mono text-xs font-semibold text-gray-700">
                          {log.method || 'GET'}
                        </span>
                      </td>
                      <td className="px-5 py-3 font-mono text-xs text-gray-700 max-w-xs truncate">
                        {log.endpoint || log.url || '—'}
                      </td>
                      <td className="px-5 py-3 text-gray-600">
                        {log.agent || log.agent_name || '—'}
                      </td>
                      <td className="px-5 py-3 text-gray-500">
                        {log.duration_ms ?? log.duration ?? '—'}
                        {(log.duration_ms !== undefined || log.duration !== undefined) && 'ms'}
                      </td>
                      <td className="px-5 py-3">
                        <span className={`px-2 py-0.5 rounded text-xs font-semibold ${colorCls}`}>
                          {sc || '—'}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between px-5 py-3 border-t border-gray-100 bg-gray-50">
          <span className="text-xs text-gray-500">Page {page}</span>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="p-1.5 rounded border border-gray-200 hover:bg-white disabled:opacity-40"
            >
              <ChevronLeft size={14} />
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={!hasMore}
              className="p-1.5 rounded border border-gray-200 hover:bg-white disabled:opacity-40"
            >
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
