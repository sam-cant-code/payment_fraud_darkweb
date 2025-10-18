import React, { useState, useMemo, useEffect } from 'react';
import MetricsGrid from '../components/dashboard/MetricsGrid';
import RiskGauge from '../components/charts/RiskGauge';
import StatusDistribution from '../components/charts/StatusDistribution';
import RiskTimeline from '../components/charts/RiskTimeline';
import TransactionsTable from '../components/dashboard/TransactionsTable';
import TransactionDetail from '../components/dashboard/TransactionDetail';
import Loader from '../components/shared/Loader';
import { AlertCircle, BarChart2 } from 'lucide-react'; // Removed RefreshCw if not used

// Added onSubmitReview prop, isLoadingReview for detail view
const DashboardPage = ({ transactions, loading, error, onSubmitReview, isLoadingReview }) => {
  const [selectedTx, setSelectedTx] = useState(null);

  // Memoize calculations for stats based on the current transactions list
  const stats = useMemo(() => {
     if (!transactions || transactions.length === 0) {
      // Return zeroed stats if no transactions
      return { totalScanned: 0, denied: 0, review: 0, approved: 0, totalValue: 0, avgScore: 0 };
    }
    const denied = transactions.filter((t) => t.final_status === 'DENY').length;
    const review = transactions.filter((t) => t.final_status === 'FLAG_FOR_REVIEW').length;
    const approved = transactions.filter((t) => t.final_status === 'APPROVE').length; // Calculate approved count
    const totalValue = transactions.reduce((acc, t) => acc + (t.value_eth || 0), 0);
    const avgScore = transactions.length > 0
        ? (transactions.reduce((acc, t) => acc + (t.final_score || 0), 0) / transactions.length)
        : 0;

    return {
        totalScanned: transactions.length, // Changed from totalFlagged
        denied,
        review,
        approved, // Include approved count
        totalValue,
        avgScore
    };
  }, [transactions]);

  // Handle transaction selection
  const handleRowClick = (tx) => {
    setSelectedTx(tx);
  };

   // Effect to update or clear selected transaction if the main list changes
   useEffect(() => {
    if (selectedTx) {
      const updatedTxInList = transactions.find(tx => tx.tx_hash === selectedTx.tx_hash);
      if (!updatedTxInList) {
        setSelectedTx(null); // Clear selection if it's no longer in the list (e.g., due to MAX_TRANSACTIONS limit)
      } else if (JSON.stringify(updatedTxInList) !== JSON.stringify(selectedTx)) {
        // If the transaction data has changed (e.g., status updated via review), update the detail view
        setSelectedTx(updatedTxInList);
      }
    }
  }, [transactions, selectedTx]);


  // Initial loading state display
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader text="Loading Initial Transaction Data..." /> {/* Updated text */}
      </div>
    );
  }

  // Error state display (if error occurred during initial load)
  if (error && transactions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <AlertCircle className="w-16 h-16 text-status-review-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Error Loading Data</h2>
        <p className="mb-4">
          Could not load initial transaction data. Please check the backend connection and status, or try running the setup again.
        </p>
        {/* Display specific error message */}
        <p className="text-sm text-slate-500">({typeof error === 'string' ? error : 'Check console for details'})</p>
      </div>
    );
  }

  // State when monitoring is active but no transactions have been scanned yet
  if (!loading && transactions.length === 0 && !error) {
     return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <BarChart2 className="w-16 h-16 text-status-approve-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">Monitoring Blockchain...</h2>
        {/* Updated message */}
        <p>Waiting for new transactions. Scanned transactions (including Approved, Flagged, and Denied) will appear here automatically.</p>
      </div>
    );
  }

  // Main dashboard display when transactions are available
  return (
    <div className="space-y-6">
      {/* 1. Key Metrics - Pass calculated stats */}
      <MetricsGrid stats={stats} />

      {/* 2. Charts */}
       <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Average Risk Score
          </h3>
          <div className="flex-grow flex items-center justify-center">
             <RiskGauge score={stats.avgScore} />
          </div>
        </div>
        <div className="lg:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200 flex flex-col">
          <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Status Distribution (Scanned)
          </h3>
           <div className="flex-grow flex items-center justify-center">
             {/* Pass the calculated stats directly */}
             <StatusDistribution data={stats} />
           </div>
        </div>
      </div>


      {/* 3. Timeline Chart */}
       <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
         <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Risk Score Timeline (Last {transactions.length} Scanned) {/* Updated Title */}
          </h3>
        {/* Ensure data passed to timeline is sorted chronologically (oldest first) */}
        <RiskTimeline data={[...transactions].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))} />
      </div>


      {/* 4. Transactions Table & Detail */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-slate-800">
                Scanned Transactions (Real-time) {/* Updated Title */}
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
          {/* Pass the specific loading state for reviews */}
          <TransactionDetail
            transaction={selectedTx}
            onClose={() => setSelectedTx(null)}
            onSubmitReview={onSubmitReview} // Pass down review function
            isLoadingReview={isLoadingReview} // Pass review loading state
          />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;