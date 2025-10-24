// frontend/src/components/layout/Sidebar.jsx
import React from 'react';
import { Link, useLocation } from 'react-router-dom'; // Import Link and useLocation
import {
  ShieldAlert,
  Database,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Wifi,
  WifiOff,
  LayoutDashboard, // Icon for Dashboard
  TestTube, // Icon for Simulation
} from 'lucide-react';

const Sidebar = ({ status, runSetup, isLoading }) => {
  const setupDisabled = status.database_ready;
  const location = useLocation(); // Hook to get current path

  // Helper to determine active link
  const isActive = (path) => location.pathname === path;

  return (
    <div className="w-64 bg-brand-dark-blue text-white flex-shrink-0 p-5 flex flex-col shadow-lg">
      <div className="flex items-center gap-3 mb-8">
        <ShieldAlert className="w-10 h-10 text-brand-blue" />
        <span className="text-2xl font-bold">AI Security</span>
      </div>

      <nav className="flex-1 space-y-6">
        {/* --- Navigation --- */}
        <div>
          <h3 className="text-xs uppercase text-slate-400 font-semibold mb-2">
            Navigation
          </h3>
          <div className="space-y-2 text-sm">
            <Link
              to="/"
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                isActive('/')
                  ? 'bg-brand-blue text-white font-medium'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" />
              <span>Dashboard</span>
            </Link>
            <Link
              to="/simulate"
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all ${
                isActive('/simulate')
                  ? 'bg-brand-blue text-white font-medium'
                  : 'text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <TestTube className="w-4 h-4" />
              <span>Submit Test Tx</span>
            </Link>
          </div>
        </div>

        {/* --- System Status (No changes) --- */}
        <div>
          <h3 className="text-xs uppercase text-slate-400 font-semibold mb-2">
            System Status
          </h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center gap-2">
              {status.websocket_connected ? (
                <Wifi className="w-4 h-4 text-status-approve-text" />
              ) : (
                <WifiOff className="w-4 h-4 text-status-deny-text" />
              )}
              <span>{status.websocket_connected ? 'Real-time Connected' : 'Disconnected'}</span>
            </div>
            <div className="flex items-center gap-2">
              {status.database_ready ? (
                <CheckCircle2 className="w-4 h-4 text-status-approve-text" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>Database Ready</span>
            </div>
            <div className="flex items-center gap-2">
              {status.listener_active ? ( // Changed this to 'listener_active'
                 <Loader2 className="w-4 h-4 text-status-approve-text animate-spin" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>{status.listener_active ? 'Listener Active' : 'Listener Inactive'}</span>
            </div>
          </div>
        </div>

        {/* --- Setup (No changes, just checking key) --- */}
        <div>
          <h3 className="text-xs uppercase text-slate-400 font-semibold mb-2">
            Setup
          </h3>
          <div className="space-y-2">
            <button
              onClick={runSetup}
              disabled={setupDisabled || isLoading}
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                setupDisabled
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-slate-700 hover:bg-brand-blue text-white'
              } ${isLoading ? 'opacity-70' : ''}`}
            >
              {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
              <Database className="w-4 h-4" />
              <span>{setupDisabled ? 'Setup Complete' : 'Run Initial Setup'}</span>
            </button>
             {!status.database_ready && (
              <p className="text-xs text-slate-400 px-1">
                Must run setup first for simulation.
              </p>
            )}
              {!status.websocket_connected && status.database_ready && !status.listener_active && (
              <p className="text-xs text-status-review-text px-1">
                Attempting real-time connection...
              </p>
            )}
          </div>
        </div>
      </nav>

      <div className="text-center text-xs text-slate-500 mt-auto">
        <p>&copy; 2025 Fraud Detection MVP</p>
        <p>v2.2.0 | Simulation Edition</p>
      </div>
    </div>
  );
};

export default Sidebar;