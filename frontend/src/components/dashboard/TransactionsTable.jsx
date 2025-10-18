import React from 'react';

const StatusBadge = ({ status }) => {
  const styles = {
    DENY: 'bg-status-deny-bg text-status-deny-text',
    FLAG_FOR_REVIEW: 'bg-status-review-bg text-status-review-text',
    APPROVE: 'bg-status-approve-bg text-status-approve-text',
  };
  return (
    <span
      className={`px-3 py-1 rounded-full text-xs font-semibold uppercase ${
        styles[status] || 'bg-slate-100 text-slate-600'
      }`}
    >
      {status.replace('_', ' ')}
    </span>
  );
};

const TransactionsTable = ({ transactions, onRowClick, selectedTxHash }) => {
  
  const shortenAddress = (addr) => `${addr.substring(0, 10)}...${addr.substring(addr.length - 8)}`;
  
  return (
    <div className="overflow-x-auto max-h-[600px] overflow-y-auto">
      <table className="min-w-full divide-y divide-slate-200">
        <thead className="bg-slate-50 sticky top-0">
          <tr>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              Tx Hash
            </th>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              From
            </th>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              To
            </th>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              Value (ETH)
            </th>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              Risk
            </th>
            <th className="py-3 px-4 text-left text-xs font-semibold text-brand-gray uppercase tracking-wider">
              Status
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-slate-200">
          {transactions.map((tx) => (
            <tr
              key={tx.tx_hash}
              onClick={() => onRowClick(tx)}
              className={`cursor-pointer transition-all ${
                selectedTxHash === tx.tx_hash
                  ? 'bg-brand-light-blue'
                  : 'hover:bg-slate-50'
              }`}
            >
              <td className="py-3 px-4 whitespace-nowrap text-sm text-slate-700 font-mono">
                {`${tx.tx_hash.substring(0, 16)}...`}
              </td>
              <td className="py-3 px-4 whitespace-nowrap text-sm text-slate-700 font-mono">
                {shortenAddress(tx.from_address)}
              </td>
              <td className="py-3 px-4 whitespace-nowrap text-sm text-slate-700 font-mono">
                {shortenAddress(tx.to_address)}
              </td>
              <td className="py-3 px-4 whitespace-nowrap text-sm text-slate-700">
                {tx.value_eth.toFixed(4)}
              </td>
              <td className="py-3 px-4 whitespace-nowrap text-sm font-bold">
                <span className={tx.final_score >= 70 ? 'text-status-deny-text' : tx.final_score >= 30 ? 'text-status-review-text' : 'text-status-approve-text'}>
                  {tx.final_score.toFixed(1)}
                </span>
              </td>
              <td className="py-3 px-4 whitespace-nowrap text-sm">
                <StatusBadge status={tx.final_status} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TransactionsTable;