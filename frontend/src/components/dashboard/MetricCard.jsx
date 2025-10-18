import React from 'react';

const MetricCard = ({ title, value, icon, delta, deltaColor }) => {
  const deltaClasses = {
    inverse: 'bg-red-100 text-red-700',
    off: 'bg-amber-100 text-amber-700',
    normal: 'bg-green-100 text-green-700',
  };
  
  return (
    <div className="bg-white p-5 rounded-lg shadow-sm border border-slate-200 transition-all hover:shadow-md">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-semibold uppercase text-brand-gray">
          {title}
        </span>
        {icon}
      </div>
      <h2 className="text-3xl font-bold text-slate-800">{value}</h2>
      {delta && (
        <span
          className={`text-xs font-medium px-2 py-0.5 rounded-full mt-2 inline-block ${
            deltaClasses[deltaColor] || deltaClasses['normal']
          }`}
        >
          {delta}
        </span>
      )}
    </div>
  );
};

export default MetricCard;