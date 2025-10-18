import React, { useState, useMemo } from 'react';
import MetricsGrid from '../components/dashboard/MetricsGrid';
import RiskGauge from '../components/charts/RiskGauge';
import StatusDistribution from '../components/charts/StatusDistribution';
import RiskTimeline from '../components/charts/RiskTimeline';
import TransactionsTable from '../components/dashboard/TransactionsTable';
import TransactionDetail from '../components/dashboard/TransactionDetail';
import Loader from '../components/shared/Loader';
import { AlertCircle, BarChart2 } from 'lucide-react';

const DashboardPage = ({ transactions, loading, error, onRefresh }) => {
  const [selectedTx, setSelectedTx] = useState(null);

  // Memoize calculations
  const stats = useMemo(() => {
    if (!transactions || transactions.length === 0) {
      return {
        totalFlagged: 0,
        denied: 0,
        review: 0,
        totalValue: 0,
        avgScore: 0,
      };
    }
    const denied = transactions.filter((t) => t.final_status === 'DENY').length;
    const review = transactions.filter((t) => t.final_status === 'FLAG_FOR_REVIEW').length;
    const totalValue = transactions.reduce((acc, t) => acc + t.value_eth, 0);
    const avgScore =
      transactions.reduce((acc, t) => acc + t.final_score, 0) /
      transactions.length;

    return {
      totalFlagged: transactions.length,
      denied,
      review,
      totalValue,
      avgScore,
    };
  }, [transactions]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Loader />
      </div>
    );
  }

  if (error && !transactions.length) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <AlertCircle className="w-16 h-16 text-status-review-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">No Data Available</h2>
        <p className="mb-4">
          Please run the simulation from the sidebar to generate and view transaction data.
        </p>
        <p className="text-sm text-slate-500">({error})</p>
      </div>
    );
  }
  
  if (transactions.length === 0) {
     return (
      <div className="flex flex-col items-center justify-center h-full text-center text-brand-gray bg-white p-8 rounded-lg shadow-md">
        <BarChart2 className="w-16 h-16 text-status-approve-text mb-4" />
        <h2 className="text-2xl font-semibold text-slate-800 mb-2">All Clear!</h2>
        <p>The simulation ran successfully and no suspicious transactions were flagged.</p>
      </div>
    );
  }

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
          <StatusDistribution data={stats} />
        </div>
      </div>
      
      {/* 3. Timeline Chart */}
      <div className="bg-white p-4 rounded-lg shadow-sm border border-slate-200">
         <h3 className="text-lg font-semibold text-slate-800 mb-2">
            Risk Score Timeline
          </h3>
        <RiskTimeline data={transactions} />
      </div>

      {/* 4. Transactions Table & Detail */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white p-4 rounded-lg shadow-sm border border-slate-200">
          <h3 className="text-lg font-semibold text-slate-800 mb-4">
            Flagged Transactions
          </h3>
          <TransactionsTable
            transactions={transactions}
            onRowClick={setSelectedTx}
            selectedTxHash={selectedTx?.tx_hash}
          />
        </div>
        <div className="xl:col-span-1">
          <TransactionDetail
            transaction={selectedTx}
            onClose={() => setSelectedTx(null)}
          />
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;