import React from 'react';
import MetricCard from './MetricCard';
import { ListChecks, ShieldOff, FileWarning, CircleDollarSign, BarChart } from 'lucide-react';

const MetricsGrid = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
      <MetricCard
        title="Total Flagged"
        value={stats.totalFlagged}
        icon={<ListChecks className="w-5 h-5 text-brand-gray" />}
        delta={`${stats.totalFlagged} transactions`}
        deltaColor="inverse"
      />
      <MetricCard
        title="Denied"
        value={stats.denied}
        icon={<ShieldOff className="w-5 h-5 text-status-deny-text" />}
        delta={stats.totalFlagged > 0 ? `${((stats.denied / stats.totalFlagged) * 100).toFixed(0)}% of total` : "0%"}
        deltaColor="inverse"
      />
      <MetricCard
        title="Under Review"
        value={stats.review}
        icon={<FileWarning className="w-5 h-5 text-status-review-text" />}
        delta={stats.totalFlagged > 0 ? `${((stats.review / stats.totalFlagged) * 100).toFixed(0)}% of total` : "0%"}
        deltaColor="off"
      />
      <MetricCard
        title="Total Value"
        value={`${stats.totalValue.toFixed(2)} ETH`}
        icon={<CircleDollarSign className="w-5 h-5 text-brand-gray" />}
      />
      <MetricCard
        title="Avg Risk"
        value={stats.avgScore.toFixed(1)}
        icon={<BarChart className="w-5 h-5 text-brand-gray" />}
        delta={
          stats.avgScore >= 70
            ? 'Critical'
            : stats.avgScore >= 50
            ? 'High'
            : 'Medium'
        }
        deltaColor={stats.avgScore >= 70 ? 'inverse' : 'off'}
      />
    </div>
  );
};

export default MetricsGrid;