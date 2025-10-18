import React from 'react';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import DashboardPage from './pages/DashboardPage';
import useDashboard from './hooks/useDashboard';
import Notification from './components/shared/Notification';

function App() {
  // Main state management hook
  const {
    status,
    transactions,
    loading,
    error,
    notification,
    runSetup,
    runSimulation,
    clearNotification,
  } = useDashboard();

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      {/* Sidebar */}
      <Sidebar
        status={status}
        runSetup={runSetup}
        runSimulation={runSimulation}
        isLoading={loading.setup || loading.simulation}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <Header />

        {/* Page Content */}
        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-slate-100 p-6 md:p-10">
          <DashboardPage
            transactions={transactions}
            loading={loading.transactions}
            error={error.transactions}
            onRefresh={runSimulation} // You might want a dedicated refresh function
          />
        </main>
      </div>
      
      {/* Notification Pop-up */}
      <Notification
        message={notification?.message}
        type={notification?.type}
        onClose={clearNotification}
      />
    </div>
  );
}

export default App;