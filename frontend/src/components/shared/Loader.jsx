import React from 'react';
import { Loader2 } from 'lucide-react';

const Loader = () => {
  return (
    <div className="flex flex-col items-center justify-center space-y-2">
      <Loader2 className="w-12 h-12 text-brand-blue animate-spin" />
      <span className="text-lg font-medium text-slate-700">Loading Data...</span>
    </div>
  );
};

export default Loader;