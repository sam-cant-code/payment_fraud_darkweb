import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, PieChart, Pie, Cell, Tooltip, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from 'recharts';
import { Shield, AlertTriangle, Activity, TrendingUp, Filter, Search, RefreshCw, Database, Play } from 'lucide-react';

// Read the API URL from Vite's environment variable
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [selectedTx, setSelectedTx] = useState(null);
  const [filters, setFilters] = useState({
    status: 'all',
    minScore: 0,
    maxScore: 150,
    search: ''
  });
  
  // State from our old app to manage API
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [status, setStatus] = useState({ database_ready: false, simulation_run: false });

  // --- API Data Fetching ---

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/status`);
      setStatus(response.data);
    } catch (err) {
      setError('Could not connect to backend API.');
    }
  };

  const fetchTransactions = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/flagged-transactions`);
      setData(response.data); // Load real data
    } catch (err) {
      if (err.response && err.response.status === 404) {
        setError('Simulation data not found. Run the simulation first.');
      } else {
        setError('Error fetching transactions.');
      }
      setData([]); // Clear data on error
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchTransactions();
  }, []); // Run once on page load

  const handleSetup = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post(`${API_BASE_URL}/api/setup`);
      alert('Setup Complete! Database is ready.');
      fetchStatus();
    } catch (err) {
      setError('Error running setup.');
    } finally {
      setLoading(false);
    }
  };

  const handleRunSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      await axios.post(`${API_BASE_URL}/api/run-simulation`);
      alert('Simulation Complete! Fetching new data...');
      fetchStatus();
      fetchTransactions(); // Fetch new data
    } catch (err) {
      setError('Error running simulation.');
    } finally {
      setLoading(false);
    }
  };

  // --- Filtering Logic (from your file) ---

  useEffect(() => {
    applyFilters();
  }, [filters, data]);

  const applyFilters = () => {
    let filtered = [...data];

    if (filters.status !== 'all') {
      filtered = filtered.filter(tx => tx.final_status === filters.status);
    }

    filtered = filtered.filter(tx =>
      parseFloat(tx.final_score) >= filters.minScore &&
      parseFloat(tx.final_score) <= filters.maxScore
    );

    if (filters.search) {
      filtered = filtered.filter(tx =>
        tx.tx_hash.toLowerCase().includes(filters.search.toLowerCase()) ||
        tx.from_address.toLowerCase().includes(filters.search.toLowerCase()) ||
        tx.to_address.toLowerCase().includes(filters.search.toLowerCase())
      );
    }

    setFilteredData(filtered);
    if (filtered.length > 0 && !selectedTx) {
      // Auto-select first tx if none is selected
      // setSelectedTx(filtered[0]);
    } else if (filtered.length === 0) {
      setSelectedTx(null);
    }
  };

  // --- Data Calculations (from your file) ---

  const stats = {
    total: filteredData.length,
    denied: filteredData.filter(tx => tx.final_status === 'DENY').length,
    flagged: filteredData.filter(tx => tx.final_status === 'FLAG_FOR_REVIEW').length,
    avgScore: filteredData.reduce((acc, tx) => acc + parseFloat(tx.final_score), 0) / filteredData.length || 0,
    totalValue: filteredData.reduce((acc, tx) => acc + parseFloat(tx.value_eth), 0)
  };

  const statusData = [
    { name: 'Denied', value: stats.denied, color: '#ef4444' },
    { name: 'Flagged', value: stats.flagged, color: '#f59e0b' },
    { name: 'Approved', value: data.length - stats.denied - stats.flagged, color: '#10b981' }
  ];

  const timelineData = filteredData.slice(0, 15).map(tx => ({
    time: new Date(tx.timestamp).toLocaleDateString(),
    score: parseFloat(tx.final_score),
    status: tx.final_status
  }));

  // --- Helper Components (from your file) ---

  const StatCard = ({ icon: Icon, label, value, delta, trend }) => (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className={`p-3 rounded-lg ${
          trend === 'up' ? 'bg-red-50' : trend === 'down' ? 'bg-green-50' : 'bg-blue-50'
        }`}>
          <Icon className={`w-6 h-6 ${
            trend === 'up' ? 'text-red-600' : trend === 'down' ? 'text-green-600' : 'text-blue-600'
          }`} />
        </div>
      </div>
      <div className="text-2xl font-bold text-gray-900 mb-1">{value}</div>
      <div className="text-sm text-gray-600">{label}</div>
      {delta && (
        <div className={`text-xs mt-2 ${trend === 'up' ? 'text-red-600' : 'text-gray-500'}`}>
          {delta}
        </div>
      )}
    </div>
  );

  const getStatusBadge = (status) => {
    const styles = {
      'DENY': 'bg-red-100 text-red-700 border-red-200',
      'FLAG_FOR_REVIEW': 'bg-yellow-100 text-yellow-700 border-yellow-200',
      'APPROVE': 'bg-green-100 text-green-700 border-green-200'
    };
    return (
      <span className={`px-3 py-1 rounded-full text-xs font-semibold border ${styles[status] || 'bg-gray-100 text-gray-700 border-gray-200'}`}>
        {status.replace('_', ' ')}
      </span>
    );
  };

  const getRiskColor = (score) => {
    const s = parseFloat(score);
    if (s >= 70) return 'text-red-600 bg-red-50';
    if (s >= 50) return 'text-yellow-600 bg-yellow-50';
    return 'text-green-600 bg-green-50';
  };

  // --- Main JSX (from your file, with API buttons added) ---

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Fraud Detection</h1>
                <p className="text-sm text-gray-600">Real-time blockchain monitoring</p>
              </div>
            </div>
            
            {/* API Control Buttons */}
            <div className="flex items-center space-x-3">
              <button
                onClick={handleSetup}
                disabled={loading}
                title="Run Setup (Builds Database)"
                className="flex items-center space-x-2 px-3 py-2 bg-gray-100 text-gray-700 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
              >
                <Database className="w-4 h-4" />
                <span className="text-sm font-medium">{status.database_ready ? 'Re-run Setup' : 'Run Setup'}</span>
              </button>
              
              <button
                onClick={handleRunSimulation}
                disabled={loading || !status.database_ready}
                title="Run Simulation"
                className="flex items-center space-x-2 px-3 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 disabled:bg-gray-400"
              >
                <Play className="w-4 h-4" />
                <span className="text-sm font-medium">Run Simulation</span>
              </button>

              <button 
                onClick={fetchTransactions}
                disabled={loading}
                title="Refresh Data"
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-5 h-5 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </div>
        {/* Error and Loading Bar */}
        {loading && <div className="h-1 bg-blue-500 animate-pulse w-full"></div>}
        {error && (
          <div className="bg-red-100 border-b border-red-300 text-red-800 text-sm font-medium text-center py-2">
            {error}
          </div>
        )}
      </header>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-6 mb-8">
          <StatCard
            icon={AlertTriangle}
            label="Flagged (in view)"
            value={stats.total}
            delta={`${stats.denied + stats.flagged} critical`}
            trend="up"
          />
          <StatCard
            icon={Shield}
            label="Denied"
            value={stats.denied}
            delta={stats.total > 0 ? `${((stats.denied / stats.total) * 100).toFixed(0)}% of total` : '0%'}
            trend="up"
          />
          <StatCard
            icon={Activity}
            label="Under Review"
            value={stats.flagged}
            delta={stats.total > 0 ? `${((stats.flagged / stats.total) * 100).toFixed(0)}% of total` : '0%'}
          />
          <StatCard
            icon={TrendingUp}
            label="Total Value (in view)"
            value={`${stats.totalValue.toFixed(2)} ETH`}
            delta={`≈ $${(stats.totalValue * 2500).toLocaleString()}`}
          />
          <StatCard
            icon={Activity}
            label="Avg Risk Score"
            value={stats.avgScore.toFixed(1)}
            delta={stats.avgScore >= 70 ? 'Critical' : 'High'}
            trend={stats.avgScore >= 70 ? 'up' : 'neutral'}
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Status Distribution (All Data)</h3>
            <div style={{ width: '100%', height: 250 }}>
              <PieChart width={300} height={250}>
                <Pie
                  data={statusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {statusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </div>
            <div className="flex justify-center space-x-4 mt-4">
              {statusData.map((item, i) => (
                <div key={i} className="flex items-center space-x-2">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                  <span className="text-xs text-gray-600">{item.name}: {item.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 lg:col-span-2">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Score Timeline (Top 15)</h3>
            <div style={{ width: '100%', height: 250 }}>
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="time" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip />
                <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} />
              </LineChart>
            </div>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 mb-6">
          <div className="flex items-center space-x-2 mb-4">
            <Filter className="w-5 h-5 text-gray-600" />
            <h3 className="text-lg font-semibold text-gray-900">Filters</h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
              <select
                value={filters.status}
                onChange={(e) => setFilters({ ...filters, status: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Statuses</option>
                <option value="DENY">Denied</option>
                <option value="FLAG_FOR_REVIEW">Flagged</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Min Score</label>
              <input
                type="number"
                value={filters.minScore}
                onChange={(e) => setFilters({ ...filters, minScore: parseInt(e.target.value) || 0 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max Score</label>
              <input
                type="number"
                value={filters.maxScore}
                onChange={(e) => setFilters({ ...filters, maxScore: parseInt(e.target.value) || 150 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
              <div className="relative">
                <Search className="absolute left-3 top-2.5 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={filters.search}
                  onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                  placeholder="Search transactions..."
                  className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-600">
            Showing <span className="font-semibold">{filteredData.length}</span> of <span className="font-semibold">{data.length}</span> transactions
          </div>
        </div>

        {/* Transactions Table */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Flagged Transactions</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Tx Hash</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">From</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">To</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Value</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Risk</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {filteredData.slice(0, 10).map((tx, idx) => (
                  <tr key={idx} className="hover:bg-gray-50 transition-colors">
                    <td className="px-6 py-4 text-sm font-mono text-gray-900">
                      {tx.tx_hash.substring(0, 16)}...
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-gray-600">
                      {tx.from_address.substring(0, 10)}...
                    </td>
                    <td className="px-6 py-4 text-sm font-mono text-gray-600">
                      {tx.to_address.substring(0, 10)}...
                    </td>
                    <td className="px-6 py-4 text-sm font-medium text-gray-900">
                      {parseFloat(tx.value_eth).toFixed(4)} ETH
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded text-sm font-bold ${getRiskColor(tx.final_score)}`}>
                        {parseFloat(tx.final_score).toFixed(1)}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      {getStatusBadge(tx.final_status)}
                    </td>
                    <td className="px-6 py-4">
                      <button
                        onClick={() => setSelectedTx(tx)}
                        className="text-blue-600 hover:text-blue-800 font-medium text-sm"
                      >
                        View Details
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Transaction Detail Modal */}
        {selectedTx && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50" onClick={() => setSelectedTx(null)}>
            <div className="bg-white rounded-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
              <div className="p-6 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-xl font-bold text-gray-900">Transaction Details</h2>
                  <button onClick={() => setSelectedTx(null)} className="text-gray-400 hover:text-gray-600">
                    <span className="text-2xl">×</span>
                  </button>
                </div>
              </div>
              <div className="p-6 space-y-6">
                <div className={`p-4 rounded-lg ${selectedTx.final_status === 'DENY' ? 'bg-red-50 border border-red-200' : 'bg-yellow-50 border border-yellow-200'}`}>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-gray-600">Risk Score</div>
                      <div className="text-3xl font-bold mt-1" style={{ color: selectedTx.final_status === 'DENY' ? '#dc2626' : '#d97706' }}>
                        {parseFloat(selectedTx.final_score).toFixed(1)} / 150
                      </div>
                    </div>
                    {getStatusBadge(selectedTx.final_status)}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">Transaction Hash</div>
                    <div className="text-sm font-mono text-gray-900 break-all bg-gray-50 p-2 rounded">{selectedTx.tx_hash}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">Timestamp</div>
                    <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">{new Date(selectedTx.timestamp).toLocaleString()}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">From Address</div>
                    <div className="text-sm font-mono text-gray-900 break-all bg-gray-50 p-2 rounded">{selectedTx.from_address}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">To Address</div>
                    <div className="text-sm font-mono text-gray-900 break-all bg-gray-50 p-2 rounded">{selectedTx.to_address}</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">Value</div>
                    <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">{parseFloat(selectedTx.value_eth).toFixed(4)} ETH (≈ ${(parseFloat(selectedTx.value_eth) * 2500).toFixed(2)})</div>
                  </div>
                  <div>
                    <div className="text-sm font-medium text-gray-600 mb-1">Gas Price</div>
                    <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">{parseFloat(selectedTx.gas_price).toFixed(2)} Gwei</div>
                  </div>
                </div>

                <div>
                  <div className="text-sm font-medium text-gray-600 mb-2">Agent Scores</div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-blue-50 border border-blue-200 p-3 rounded-lg">
                      <div className="text-xs text-blue-600 font-medium">Agent 1 (Threat Intel)</div>
                      <div className="text-xl font-bold text-blue-900 mt-1">{parseFloat(selectedTx.agent_1_score).toFixed(1)}</div>
                    </div>
                    <div className="bg-purple-50 border border-purple-200 p-3 rounded-lg">
                      <div className="text-xs text-purple-600 font-medium">Agent 2 (Behavioral)</div>
                      <div className="text-xl font-bold text-purple-900 mt-1">{parseFloat(selectedTx.agent_2_score).toFixed(1)}</div>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-sm font-medium text-gray-600 mb-2">Flagging Reasons</div>
                  <div className="space-y-2">
                    {selectedTx.reasons.split(' | ').map((reason, idx) => (
                      <div key={idx} className={`p-3 rounded-lg border ${
                        reason.includes('Dark Web') || reason.includes('Mixer')
                          ? 'bg-red-50 border-red-200'
                          : 'bg-yellow-50 border-yellow-200'
                      }`}>
                        <div className="flex items-start space-x-2">
                          <AlertTriangle className={`w-4 h-4 mt-0.5 ${
                            reason.includes('Dark Web') || reason.includes('Mixer')
                              ? 'text-red-600'
                              : 'text-yellow-600'
                          }`} />
                          <span className="text-sm text-gray-900">{reason}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;