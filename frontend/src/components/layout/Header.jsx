import React from 'react';
import { ShieldCheck } from 'lucide-react';

const Header = () => {
  return (
    <header className="flex-shrink-0 bg-white h-16 md:h-20 flex items-center justify-between px-6 shadow-sm border-b border-slate-200">
      <div>
        <h1 className="text-xl md:text-2xl font-bold text-slate-800">
          Fraud Detection Dashboard
        </h1>
        <p className="text-sm text-brand-gray hidden md:block">
          Real-time transaction monitoring powered by AI
        </p>
      </div>
      <div className="flex items-center gap-2 text-status-approve-text">
        <ShieldCheck className="w-6 h-6" />
        <span className="font-semibold text-slate-700">System Secure</span>
      </div>
    </header>
  );
};

export default Header;