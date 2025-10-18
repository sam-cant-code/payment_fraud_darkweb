import React from 'react';
import {
  ShieldAlert,
  Database,
  PlayCircle, // Keep icon for visual consistency if needed, or remove
  CheckCircle2,
  AlertCircle,
  Loader2,
  Wifi, // Icon for connection status
  WifiOff // Icon for disconnected status
} from 'lucide-react';

// Removed runSimulation from props
const Sidebar = ({ status, runSetup, isLoading }) => {
  const setupDisabled = status.database_ready;
  // Simulation button is now gone or always disabled

  return (
    <div className="w-64 bg-brand-dark-blue text-white flex-shrink-0 p-5 flex flex-col shadow-lg">
      <div className="flex items-center gap-3 mb-8">
        <ShieldAlert className="w-10 h-10 text-brand-blue" />
        <span className="text-2xl font-bold">AI Security</span>
      </div>

      <nav className="flex-1 space-y-4">
        <div>
          <h3 className="text-xs uppercase text-slate-400 font-semibold mb-2">
            System Status
          </h3>
          <div className="space-y-2 text-sm">
             {/* WebSocket Connection Status */}
             <div className="flex items-center gap-2">
              {status.websocket_connected ? (
                <Wifi className="w-4 h-4 text-status-approve-text" />
              ) : (
                <WifiOff className="w-4 h-4 text-status-deny-text" />
              )}
              <span>{status.websocket_connected ? 'Real-time Connected' : 'Disconnected'}</span>
            </div>
             {/* Database Status */}
            <div className="flex items-center gap-2">
              {status.database_ready ? (
                <CheckCircle2 className="w-4 h-4 text-status-approve-text" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>Database Ready</span>
            </div>
            {/* Simulation Running Status */}
            <div className="flex items-center gap-2">
              {status.simulation_running ? (
                 <Loader2 className="w-4 h-4 text-status-approve-text animate-spin" />
                // <CheckCircle2 className="w-4 h-4 text-status-approve-text" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>{status.simulation_running ? 'Simulation Active' : 'Simulation Inactive'}</span>
            </div>
          </div>
        </div>

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
            {/* Simulation Button Removed */}
            {/* <button
              // onClick={runSimulation} // Removed onClick
              disabled={true} // Always disabled
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all bg-slate-700 text-slate-500 cursor-not-allowed`}
            >
              <PlayCircle className="w-4 h-4" />
              <span>Simulation Runs Automatically</span>
            </button> */}
             {!status.database_ready && (
              <p className="text-xs text-slate-400 px-1">
                Must run setup first for simulation.
              </p>
            )}
              {!status.websocket_connected && status.database_ready && (
              <p className="text-xs text-status-review-text px-1">
                Attempting real-time connection...
              </p>
            )}
          </div>
        </div>
      </nav>

      <div className="text-center text-xs text-slate-500 mt-auto">
        <p>&copy; 2025 Fraud Detection MVP</p>
        <p>v2.1.0 | Real-time Edition</p>
      </div>
    </div>
  );
};

export default Sidebar;