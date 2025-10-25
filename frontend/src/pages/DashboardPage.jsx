// frontend/src/pages/DashboardPage.jsx
import React, { useState, useMemo, useEffect } from 'react';
import MetricsGrid from '../components/dashboard/MetricsGrid';
import RiskGauge from '../components/charts/RiskGauge';
import StatusDistribution from '../components/charts/StatusDistribution';
import RiskTimeline from '../components/charts/RiskTimeline';
import TransactionsTable from '../components/dashboard/TransactionsTable';
import TransactionDetail from '../components/dashboard/TransactionDetail';
import Loader from '../components/shared/Loader';
import { AlertCircle, BarChart2, Filter, XCircle } from 'lucide-react';
import { useTransactionStore } from '../store/transactionStore';
import NetworkGraph from '../components/charts/NetworkGraph';
import Agent3Stats from '../components/dashboard/Agent3Stats';

const DashboardPage = ({ onSubmitReview, isLoadingReview }) => {
  // --- Zustand State ---
  const {
    allTransactions,
    filteredTransactions,
    filters,
    setFilter,
    clearFilters
  } = useTransactionStore();

  // --- Local UI State ---
  const [selectedTx, setSelectedTx] = useState(null);
  const [showFilters, setShowFilters] = useState(false);
  
  const isLoadingInitial = useTransactionStore(state => 
    state.allTransactions.length === 0 && 
    !state.filters.sender && 
    !state.filters.receiver
  );

  // ✅ FIXED: Correct stats calculation
  const stats = useMemo(() => {
    if (!allTransactions || allTransactions.length === 0) {
      return { 
        totalScanned: 0, 
        totalFlagged: 0, // NEW: Only DENY + REVIEW
        denied: 0, 
        review: 0, 
        approved: 0, 
        totalValue: 0, 
        avgScore: 0 
      };
    }

    const denied = allTransactions.filter((t) => t.final_status === 'DENY').length;
    const review = allTransactions.filter((t) => t.final_status === 'FLAG_FOR_REVIEW').length;
    const approved = allTransactions.filter((t) => t.final_status === 'APPROVE').length;
    
    // ✅ FIXED: Total flagged = DENY + REVIEW only (not including APPROVE)
    const totalFlagged = denied + review;
    
    const totalValue = allTransactions.reduce((acc, t) => acc + (t.value_eth || 0), 0);
    
    // ✅ FIXED: Calculate average score across ALL transactions
    const avgScore = allTransactions.length > 0
      ? allTransactions.reduce((acc, t) => acc + (t.final_score || 0), 0) / allTransactions.length
      : 0;

    return {
      totalScanned: allTransactions.length, // Total scanned
      totalFlagged, // Only DENY + REVIEW
      denied,
      review,
      approved,
      totalValue,
      avgScore
    };
  }, [allTransactions]);

  // Handle transaction selection
  const handleRowClick = (tx) => {
    setSelectedTx(tx);
  };

  // Effect to update selected transaction
  useEffect(() => {
    if (selectedTx) {
      const updatedTxInList = filteredTransactions.find(tx => tx.tx_hash === selectedTx.tx_hash);
      if (!updatedTxInList) {
        setSelectedTx(null);
      } else if (JSON.stringify(updatedTxInList) !== JSON.stringify(selectedTx)) {
        setSelectedTx(updatedTxInList);
      }
    } else if (filteredTransactions.length === 0 && selectedTx) {
      setSelectedTx(null);
    }
  }, [filteredTransactions, selectedTx]);

  // Handle filter input changes
  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    if (name === 'minValue' || name === 'maxValue' || name === 'minRisk' || name === 'maxRisk') {
      const numValue = value === '' ? null : parseFloat(value);
      if (!isNaN(numValue) || numValue === null) {
        setFilter(name, numValue);
      }
    } else {
      setFilter(name, value);
    }
  };

  // Render Loading state
  if (isLoadingInitial && allTransactions.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader text="Loading Initial Transaction Data..." />
      </div>
    );
  }

  // Render No Transactions state
  if (allTransactions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <BarChart2 className="w-16 h-16 text-status-approve-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Monitoring Blockchain...</h2>
        <p>Waiting for new transactions. Scanned transactions will appear here automatically.</p>
      </div>
    );
  }

  // Main dashboard display
  return (
    <div className="space-y-6">
      {/* 1. Key Metrics - ✅ Now shows correct stats */}
      <MetricsGrid stats={stats} />

      {/* 2. Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Average Risk Score
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            Across all {stats.totalScanned} scanned transactions
          </p>
          <div className="flex-grow flex items-center justify-center">
            <RiskGauge score={stats.avgScore} />
          </div>
        </div>
        <div className="lg:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Status Distribution
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            {stats.totalScanned} total transactions scanned
          </p>
          <div className="flex-grow flex items-center justify-center">
            <StatusDistribution data={stats} />
          </div>
        </div>
      </div>

      {/* 3. Timeline Chart */}
      <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
        <h3 className="text-lg font-semibold text-slate-800 mb-2">
          Risk Score Timeline
        </h3>
        <p className="text-xs text-slate-500 mb-2">
          Showing {filteredTransactions.length} filtered transactions
        </p>
        <RiskTimeline 
          data={[...filteredTransactions].sort((a, b) => 
            new Date(a.timestamp) - new Date(b.timestamp)
          )} 
        />
      </div>

      {/* 4. Network Analysis Section */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Agent 3 Stats */}
        <div className="xl:col-span-1 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <Agent3Stats transactions={allTransactions} />
        </div>

        {/* Network Graph */}
        <div className="xl:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Transaction Network Graph
          </h3>
          <p className="text-xs text-slate-500 mb-2">
            Visualizing recent transaction flows and network patterns
          </p>
          <NetworkGraph 
            transactions={filteredTransactions} 
            maxTransactions={50} 
          />
        </div>
      </div>

      {/* 5. Transactions Table & Detail */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          {/* Filter Panel */}
          {showFilters && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pb-4 mb-4 border-b border-slate-200">
              <input
                type="text"
                name="sender"
                placeholder="Filter by Sender (address)"
                value={filters.sender}
                onChange={handleFilterChange}
                className="p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
              />
              <input
                type="text"
                name="receiver"
                placeholder="Filter by Receiver (address)"
                value={filters.receiver}
                onChange={handleFilterChange}
                className="p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
              />
              <input
                type="text"
                name="txHash"
                placeholder="Filter by Tx Hash"
                value={filters.txHash}
                onChange={handleFilterChange}
                className="p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
              />
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  name="minValue"
                  placeholder="Min Value (ETH)"
                  value={filters.minValue ?? ''}
                  onChange={handleFilterChange}
                  min="0"
                  step="0.01"
                  className="w-1/2 p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                />
                <span>-</span>
                <input
                  type="number"
                  name="maxValue"
                  placeholder="Max Value (ETH)"
                  value={filters.maxValue ?? ''}
                  onChange={handleFilterChange}
                  min="0"
                  step="0.01"
                  className="w-1/2 p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  name="minRisk"
                  placeholder="Min Risk"
                  value={filters.minRisk ?? ''}
                  onChange={handleFilterChange}
                  min="0"
                  max="150"
                  step="1"
                  className="w-1/2 p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                />
                <span>-</span>
                <input
                  type="number"
                  name="maxRisk"
                  placeholder="Max Risk"
                  value={filters.maxRisk ?? ''}
                  onChange={handleFilterChange}
                  min="0"
                  max="150"
                  step="1"
                  className="w-1/2 p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                />
              </div>
              <button
                onClick={clearFilters}
                className="col-span-1 flex items-center justify-center gap-1 p-2 bg-slate-100 text-slate-600 rounded-md hover:bg-slate-200 text-sm font-medium"
              >
                <XCircle className="w-4 h-4" /> Clear Filters
              </button>
            </div>
          )}

          {/* Table */}
          <div>
            {filteredTransactions.length > 0 ? (
              <TransactionsTable
                transactions={filteredTransactions}
                onRowClick={handleRowClick}
                selectedTxHash={selectedTx?.tx_hash}
                onToggleFilters={() => setShowFilters(!showFilters)}
                isFilterOpen={showFilters}
              />
            ) : (
              <div>
                <TransactionsTable
                  transactions={[]}
                  onRowClick={() => {}}
                  selectedTxHash={null}
                  onToggleFilters={() => setShowFilters(!showFilters)}
                  isFilterOpen={showFilters}
                />
                <p className="text-center text-brand-gray py-8">
                  No transactions match the current filters.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Transaction Detail Panel */}
        <div className="xl:col-span-1">
          <TransactionDetail
            transaction={selectedTx}
            onClose={() => setSelectedTx(null)}
            onSubmitReview={onSubmitReview}
            isLoadingReview={isLoadingReview}
          />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;