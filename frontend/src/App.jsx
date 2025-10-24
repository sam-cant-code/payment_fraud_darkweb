// frontend/src/App.jsx
import React from 'react';
import { Routes, Route } from 'react-router-dom'; // Import routing components
import Sidebar from './components/layout/Sidebar';
import Header from './components/layout/Header';
import DashboardPage from './pages/DashboardPage';
import SimulationPage from './pages/SimulationPage'; // Import the new page
import useDashboard from './hooks/useDashboard';
import Notification from './components/shared/Notification';

function App() {
  const {
    status,
    loading,
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
        isLoading={loading.setup}
      />

      <div className="flex-1 flex flex-col overflow-hidden">
        <Header connectionStatus={status.websocket_connected} />

        <main className="flex-1 overflow-x-hidden overflow-y-auto bg-slate-100 p-6 md:p-10">
          {/* Define your application's routes here */}
          <Routes>
            <Route
              path="/"
              element={
                <DashboardPage
                  onSubmitReview={submitTxReview}
                  isLoadingReview={loading.review}
                />
              }
            />
            <Route path="/simulate" element={<SimulationPage />} />
          </Routes>
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