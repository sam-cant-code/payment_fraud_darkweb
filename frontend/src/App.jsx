import React from 'react';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import DashboardPage from './pages/DashboardPage';
import useDashboard from './hooks/useDashboard';
import Notification from './components/shared/Notification';

function App() {
  const {
    status,
    transactions,
    loading,
    error,
    notification,
    runSetup,
    // runSimulation, // Removed
    clearNotification,
    submitTxReview, // Pass review function down
    // ... potentially pass other new functions if needed by children
  } = useDashboard();

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      <Sidebar
        status={status}
        runSetup={runSetup}
        // Pass only the setup loading state
        isLoading={loading.setup}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header connectionStatus={status.websocket_connected} /> {/* Optional: Pass status to Header */}

        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-slate-100 p-6 md:p-10">
          <DashboardPage
            transactions={transactions}
            // Pass initial transaction loading state
            loading={loading.transactions && transactions.length === 0} // Show loader only on initial load or if empty
            error={error.transactions}
            // onRefresh={fetchTransactions} // fetchTransactions now only loads initial, maybe remove refresh?
            onSubmitReview={submitTxReview} // Pass down the review function
          />
        </main>
      </div>

      <Notification
        message={notification?.message}
        type={notification?.type}
        onClose={clearNotification}
      />
    </div>
  );
}

export default App;