import React from 'react';
import {
  ShieldAlert,
  Database,
  PlayCircle,
  CheckCircle2,
  AlertCircle,
  Loader2
} from 'lucide-react';

const Sidebar = ({ status, runSetup, runSimulation, isLoading }) => {
  const setupDisabled = status.database_ready;
  const simDisabled = !status.database_ready;

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
            <div className="flex items-center gap-2">
              {status.database_ready ? (
                <CheckCircle2 className="w-4 h-4 text-status-approve-text" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>Database Ready</span>
            </div>
            <div className="flex items-center gap-2">
              {status.simulation_run ? (
                <CheckCircle2 className="w-4 h-4 text-status-approve-text" />
              ) : (
                <AlertCircle className="w-4 h-4 text-status-review-text" />
              )}
              <span>Simulation Run</span>
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-xs uppercase text-slate-400 font-semibold mb-2">
            Actions
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
              <span>{setupDisabled ? 'Setup Complete' : '1. Run Setup'}</span>
            </button>
            <button
              onClick={runSimulation}
              disabled={simDisabled || isLoading}
              className={`w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                simDisabled
                  ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
                  : 'bg-brand-blue hover:bg-blue-500 text-white'
              } ${isLoading ? 'opacity-70' : ''}`}
            >
              {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
              <PlayCircle className="w-4 h-4" />
              <span>{status.simulation_run ? '2. Re-run Simulation' : '2. Run Simulation'}</span>
            </button>
            {simDisabled && (
              <p className="text-xs text-slate-400 px-1">
                Must run setup first.
              </p>
            )}
          </div>
        </div>
      </nav>

      <div className="text-center text-xs text-slate-500 mt-auto">
        <p>&copy; 2025 Fraud Detection MVP</p>
        <p>v2.0.0 | React Edition</p>
      </div>
    </div>
  );
};

export default Sidebar;