import React, { useState, useMemo, useEffect } from 'react'; // <-- Added useEffect here
import MetricsGrid from '../components/dashboard/MetricsGrid';
import RiskGauge from '../components/charts/RiskGauge';
import StatusDistribution from '../components/charts/StatusDistribution';
import RiskTimeline from '../components/charts/RiskTimeline';
import TransactionsTable from '../components/dashboard/TransactionsTable';
import TransactionDetail from '../components/dashboard/TransactionDetail';
import Loader from '../components/shared/Loader';
import { AlertCircle, BarChart2, RefreshCw } from 'lucide-react';

// Added onSubmitReview prop
const DashboardPage = ({ transactions, loading, error, onSubmitReview }) => {
  const [selectedTx, setSelectedTx] = useState(null);

  // Memoize calculations (remains the same)
  const stats = useMemo(() => {
     if (!transactions || transactions.length === 0) {
      return { totalFlagged: 0, denied: 0, review: 0, totalValue: 0, avgScore: 0 };
    }
    const denied = transactions.filter((t) => t.final_status === 'DENY').length;
    const review = transactions.filter((t) => t.final_status === 'FLAG_FOR_REVIEW').length;
    const totalValue = transactions.reduce((acc, t) => acc + t.value_eth, 0);
    const avgScore = transactions.length > 0 ? (transactions.reduce((acc, t) => acc + t.final_score, 0) / transactions.length) : 0;
    return { totalFlagged: transactions.length, denied, review, totalValue, avgScore };
  }, [transactions]);

  // Handle transaction selection, clear detail if selected is removed/updated
  const handleRowClick = (tx) => {
    setSelectedTx(tx);
  };

   // Check if the currently selected transaction still exists in the updated list
  useEffect(() => { // <-- This is where useEffect was used without import
    if (selectedTx && !transactions.find(tx => tx.tx_hash === selectedTx.tx_hash)) {
      setSelectedTx(null); // Clear selection if it's gone
    }
     // Optionally update selectedTx if its details (like status) changed
     else if (selectedTx) {
         const updatedTx = transactions.find(tx => tx.tx_hash === selectedTx.tx_hash);
         // Check if updatedTx exists and if it's actually different before updating state
         if (updatedTx && JSON.stringify(updatedTx) !== JSON.stringify(selectedTx)) {
             setSelectedTx(updatedTx);
         }
     }
  }, [transactions, selectedTx]);


  // Initial loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        {/* Pass text prop to Loader */}
        <Loader text="Loading Initial Data..." />
      </div>
    );
  }

  // Error state *after* initial load attempt (if transactions remain empty)
  if (error && transactions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <AlertCircle className="w-16 h-16 text-status-review-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Error Loading Data</h2>
        <p className="mb-4">
          Could not load initial transaction data. Please check the backend status or try setting up again.
        </p>
        <p className="text-sm text-slate-500">({error})</p>
      </div>
    );
  }

  // No transactions flagged yet (could be initial state or after running)
  if (transactions.length === 0) {
     return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <BarChart2 className="w-16 h-16 text-status-approve-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Monitoring Active</h2>
        <p>No suspicious transactions flagged yet. New flagged transactions will appear here automatically.</p>
        {/* Show error here too, if applicable */}
        {error && <p className="text-sm text-slate-500 mt-2">({error})</p>}
      </div>
    );
  }

  // Main dashboard display
  return (
    <div className="space-y-6">
      {/* 1. Key Metrics */}
      <MetricsGrid stats={stats} />

      {/* 2. Charts */}
       <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Average Risk Score
          </h3>
          <RiskGauge score={stats.avgScore} />
        </div>
        <div className="lg:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Status Distribution
          </h3>
          {/* Pass the calculated stats directly */}
          <StatusDistribution data={stats} />
        </div>
      </div>


      {/* 3. Timeline Chart */}
       <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
         <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Risk Score Timeline (Last {transactions.length} Flagged)
          </h3>
        {/* Ensure data passed to timeline is sorted chronologically (oldest first) */}
        <RiskTimeline data={[...transactions].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))} />
      </div>


      {/* 4. Transactions Table & Detail */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-slate-800">
                Flagged Transactions (Real-time)
              </h3>
              {/* Optional: Manual refresh button (less needed now) */}
              {/* <button onClick={onRefresh} className="text-sm text-brand-blue hover:underline p-1"><RefreshCw className="w-4 h-4 inline mr-1"/> Refresh List</button> */}
          </div>
          <TransactionsTable
            transactions={transactions} // Already sorted newest first by hook
            onRowClick={handleRowClick}
            selectedTxHash={selectedTx?.tx_hash}
          />
        </div>
        <div className="xl:col-span-1">
          <TransactionDetail
            transaction={selectedTx}
            onClose={() => setSelectedTx(null)}
            onSubmitReview={onSubmitReview} // Pass down review function
            // Pass the specific loading state for reviews if available from hook
            // isLoadingReview={loading.review}
          />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;