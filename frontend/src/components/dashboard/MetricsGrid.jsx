import React from 'react';
import MetricCard from './MetricCard';
import { ListChecks, ShieldOff, FileWarning, CircleDollarSign, BarChart, Scan } from 'lucide-react';

const MetricsGrid = ({ stats }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-4">
      {/* Total Scanned */}
      <MetricCard
        title="Total Scanned"
        value={stats.totalScanned}
        icon={<Scan className="w-5 h-5 text-brand-blue" />}
        delta={`${stats.totalScanned} transactions`}
        deltaColor="normal"
      />

      {/* Total Flagged (DENY + REVIEW only) */}
      <MetricCard
        title="Total Flagged"
        value={stats.totalFlagged}
        icon={<ListChecks className="w-5 h-5 text-status-review-text" />}
        delta={stats.totalScanned > 0 ? `${((stats.totalFlagged / stats.totalScanned) * 100).toFixed(1)}% of scanned` : "0%"}
        deltaColor="inverse"
      />

      {/* Denied */}
      <MetricCard
        title="Denied"
        value={stats.denied}
        icon={<ShieldOff className="w-5 h-5 text-status-deny-text" />}
        delta={stats.totalScanned > 0 ? `${((stats.denied / stats.totalScanned) * 100).toFixed(1)}% of scanned` : "0%"}
        deltaColor="inverse"
      />

      {/* Under Review */}
      <MetricCard
        title="Under Review"
        value={stats.review}
        icon={<FileWarning className="w-5 h-5 text-status-review-text" />}
        delta={stats.totalScanned > 0 ? `${((stats.review / stats.totalScanned) * 100).toFixed(1)}% of scanned` : "0%"}
        deltaColor="off"
      />

      {/* Total Value */}
      <MetricCard
        title="Total Value"
        value={`${stats.totalValue.toFixed(2)} ETH`}
        icon={<CircleDollarSign className="w-5 h-5 text-brand-gray" />}
        delta={stats.totalValue > 0 ? `≈ $${(stats.totalValue * 2500).toLocaleString()}` : "$0"}
        deltaColor="normal"
      />

      {/* Avg Risk */}
      <MetricCard
        title="Avg Risk"
        value={stats.avgScore.toFixed(1)}
        icon={<BarChart className="w-5 h-5 text-brand-gray" />}
        delta={
          stats.avgScore >= 70
            ? 'Critical'
            : stats.avgScore >= 45
            ? 'High'
            : stats.avgScore >= 20
            ? 'Medium'
            : 'Low'
        }
        deltaColor={stats.avgScore >= 70 ? 'inverse' : stats.avgScore >= 45 ? 'off' : 'normal'}
      />
    </div>
  );
};

export default MetricsGrid;