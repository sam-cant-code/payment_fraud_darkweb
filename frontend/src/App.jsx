// frontend/src/App.jsx
// No changes needed here based on the provided code,
// as DashboardPage now gets its data from the Zustand store.
// Ensure useDashboard still returns necessary props like status, loading flags, error flags, runSetup, submitTxReview etc.
import React from 'react';
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import DashboardPage from './pages/DashboardPage';
import useDashboard from './hooks/useDashboard'; // Keep using the hook for status, actions etc.
import Notification from './components/shared/Notification';

function App() {
  // Destructure what's still needed from the hook (status, actions, local loading/error)
  const {
    status,
    loading, // Keep loading flags for setup, review etc.
    error,
    notification,
    runSetup,
    clearNotification,
    submitTxReview,
  } = useDashboard();

  return (
    <div className="flex h-screen bg-gray-100 font-sans">
      <Sidebar
        status={status}
        runSetup={runSetup}
        isLoading={loading.setup} // Pass specific loading flag
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header connectionStatus={status.websocket_connected} />

        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-slate-100 p-6 md:p-10">
          {/* DashboardPage now uses Zustand internally for transactions */}
          <DashboardPage
            // Pass down actions and relevant loading/error states if needed by children
            onSubmitReview={submitTxReview}
            isLoadingReview={loading.review} // Pass the specific review loading state
            // No need to pass transactions, loading.transactions, error.transactions anymore
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