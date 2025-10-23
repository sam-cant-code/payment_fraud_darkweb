// frontend/src/components/dashboard/TransactionsTable.jsx
import React from 'react';
import { format, parseISO } from 'date-fns';
import { Filter, ChevronDown } from 'lucide-react'; // Import icons

// Helper function for status styling
const getStatusClass = (status) => {
  switch (status) {
    case 'DENY':
      return 'bg-status-deny-bg text-status-deny-text';
    case 'FLAG_FOR_REVIEW':
      return 'bg-status-review-bg text-status-review-text';
    case 'APPROVE':
      return 'bg-status-approve-bg text-status-approve-text';
    default:
      return 'bg-gray-200 text-gray-800';
  }
};

// Helper to shorten addresses
const shortenAddress = (address) => {
    if (!address) return 'N/A';
    return `${address.substring(0, 6)}...${address.substring(address.length - 4)}`;
}

const TransactionsTable = ({
  transactions,
  onRowClick,
  selectedTxHash,
  onToggleFilters, // New prop: function to toggle filters
  isFilterOpen     // New prop: boolean to show chevron state
}) => {
  return (
    <div> {/* Wrapper div */}

      {/* --- NEW HEADER SECTION --- */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold text-slate-800">
          Filtered Transactions ({transactions.length})
        </h3>
        <button
          onClick={onToggleFilters}
          title="Toggle Filters"
          className="flex items-center gap-2 p-2 text-sm font-medium text-brand-blue rounded-md hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-brand-blue"
        >
          <Filter className="w-4 h-4" />
          <span>Filter</span>
          <ChevronDown
            className={`w-5 h-5 transition-transform ${
              isFilterOpen ? 'rotate-180' : ''
            }`}
          />
        </button>
      </div>
      {/* --- END NEW HEADER SECTION --- */}


      {/* Existing Table Code */}
      <div className="overflow-x-auto w-full">
        <table className="min-w-full divide-y divide-slate-200">
          <thead className="bg-slate-50">
            <tr>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Status
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Tx Hash
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Risk Score
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Timestamp
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                Value (ETH)
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                From
              </th>
              <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                To
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-slate-200">
            {transactions.map((tx) => (
              <tr
                key={tx.tx_hash}
                onClick={() => onRowClick(tx)}
                className={`cursor-pointer hover:bg-slate-50 ${
                  selectedTxHash === tx.tx_hash ? 'bg-brand-blue/10' : ''
                }`}
              >
                <td className="px-4 py-3 whitespace-nowrap">
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-semibold ${getStatusClass(
                      tx.final_status
                    )}`}
                  >
                    {tx.final_status.replace('_', ' ')}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-900 font-mono" title={tx.tx_hash}>
                  {shortenAddress(tx.tx_hash)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-slate-900">
                  {tx.final_score.toFixed(0)}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-500">
                  {format(parseISO(tx.timestamp), 'MMM d, yyyy h:mm:ss a')}
                </td>
                <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-900 font-medium">
                  {tx.value_eth.toFixed(6)}
                </td>
                 <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-900 font-mono" title={tx.from_address}>
                  {shortenAddress(tx.from_address)}
                </td>
                 <td className="px-4 py-3 whitespace-nowrap text-sm text-slate-900 font-mono" title={tx.to_address}>
                  {shortenAddress(tx.to_address)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default TransactionsTable;