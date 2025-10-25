// frontend/src/components/dashboard/Agent3Stats.jsx
import React from 'react';
import { Network, TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';

/**
 * Agent3Stats Component
 * Displays statistics extracted from Agent 3's reasons in transactions
 * Shows real-time pattern detection from the backend's NetworkX graph
 */
const Agent3Stats = ({ transactions }) => {
  // Parse Agent 3 reasons from recent transactions
  const parseAgent3Data = () => {
    const stats = {
      circularPatterns: 0,
      fanOutDetected: 0,
      fanInDetected: 0,
      totalAgent3Alerts: 0,
      recentAlerts: [],
    };

    // Look through recent transactions for Agent 3 patterns
    transactions.forEach((tx) => {
      const reasons = tx.reasons || '';
      const agent3Score = tx.agent_3_score || 0;

      if (agent3Score > 0) {
        stats.totalAgent3Alerts++;

        // Check for specific patterns
        if (reasons.toLowerCase().includes('circular')) {
          stats.circularPatterns++;
          stats.recentAlerts.push({
            type: 'Circular',
            tx: tx.tx_hash,
            score: agent3Score,
            timestamp: tx.timestamp,
          });
        }
        if (reasons.toLowerCase().includes('fan-out') || reasons.toLowerCase().includes('smurfing')) {
          stats.fanOutDetected++;
          stats.recentAlerts.push({
            type: 'Fan-Out',
            tx: tx.tx_hash,
            score: agent3Score,
            timestamp: tx.timestamp,
          });
        }
        if (reasons.toLowerCase().includes('fan-in') || reasons.toLowerCase().includes('collection')) {
          stats.fanInDetected++;
          stats.recentAlerts.push({
            type: 'Fan-In',
            tx: tx.tx_hash,
            score: agent3Score,
            timestamp: tx.timestamp,
          });
        }
      }
    });

    // Sort alerts by most recent
    stats.recentAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
    stats.recentAlerts = stats.recentAlerts.slice(0, 5); // Keep top 5

    return stats;
  };

  const stats = parseAgent3Data();

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-5 h-5 text-brand-blue" />
          <h4 className="text-sm font-semibold text-slate-800 uppercase">
            Agent 3: Network Analysis
          </h4>
        </div>
        <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
          Real-time Graph
        </span>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-3 rounded-lg border border-blue-200">
          <div className="flex items-center justify-between mb-1">
            <Network className="w-4 h-4 text-blue-600" />
            <span className="text-2xl font-bold text-blue-900">{stats.totalAgent3Alerts}</span>
          </div>
          <p className="text-xs text-blue-700 font-medium">Total Alerts</p>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-3 rounded-lg border border-purple-200">
          <div className="flex items-center justify-between mb-1">
            <RefreshCw className="w-4 h-4 text-purple-600" />
            <span className="text-2xl font-bold text-purple-900">{stats.circularPatterns}</span>
          </div>
          <p className="text-xs text-purple-700 font-medium">Circular Flow</p>
        </div>

        <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-3 rounded-lg border border-orange-200">
          <div className="flex items-center justify-between mb-1">
            <TrendingUp className="w-4 h-4 text-orange-600" />
            <span className="text-2xl font-bold text-orange-900">{stats.fanOutDetected}</span>
          </div>
          <p className="text-xs text-orange-700 font-medium">Fan-Out (Smurf)</p>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 p-3 rounded-lg border border-green-200">
          <div className="flex items-center justify-between mb-1">
            <TrendingDown className="w-4 h-4 text-green-600" />
            <span className="text-2xl font-bold text-green-900">{stats.fanInDetected}</span>
          </div>
          <p className="text-xs text-green-700 font-medium">Fan-In (Collect)</p>
        </div>
      </div>

      {/* Recent Alerts */}
      {stats.recentAlerts.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-slate-700 uppercase mb-2">
            Recent Network Alerts
          </h5>
          <div className="space-y-2">
            {stats.recentAlerts.map((alert, i) => (
              <div
                key={i}
                className="flex items-center justify-between p-2 bg-slate-50 rounded border border-slate-200 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2 py-0.5 rounded font-semibold ${
                      alert.type === 'Circular'
                        ? 'bg-purple-100 text-purple-700'
                        : alert.type === 'Fan-Out'
                        ? 'bg-orange-100 text-orange-700'
                        : 'bg-green-100 text-green-700'
                    }`}
                  >
                    {alert.type}
                  </span>
                  <span className="font-mono text-slate-600">
                    {alert.tx.slice(0, 10)}...
                  </span>
                </div>
                <span className="text-slate-500">
                  Score: <span className="font-semibold text-slate-700">{alert.score}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info Note */}
      <div className="bg-blue-50 border border-blue-200 rounded-md p-3">
        <p className="text-xs text-blue-800">
          <strong>🔍 How it works:</strong> Agent 3 maintains a live in-memory graph 
          (using NetworkX) of the last ~20,000 transactions. It detects suspicious 
          patterns like circular money flows, mass distribution (smurfing), and 
          collection points in real-time.
        </p>
      </div>
    </div>
  );
};

export default Agent3Stats;