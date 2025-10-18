import React from 'react';
import { X, AlertTriangle } from 'lucide-react';
import RiskGauge from '../charts/RiskGauge';

const TransactionDetail = ({ transaction, onClose }) => {
  if (!transaction) {
    return (
      <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 h-full flex flex-col items-center justify-center text-center">
        <AlertTriangle className="w-12 h-12 text-slate-400 mb-4" />
        <h3 className="text-lg font-semibold text-slate-700">
          No Transaction Selected
        </h3>
        <p className="text-sm text-brand-gray">
          Click on a row in the table to see details.
        </p>
      </div>
    );
  }

  const {
    tx_hash,
    from_address,
    to_address,
    value_eth,
    gas_price,
    timestamp,
    final_status,
    final_score,
    reasons,
    agent_1_score,
    agent_2_score,
  } = transaction;

  const reasonsList = reasons.split(' | ');
  const isDeny = final_status === 'DENY';
  
  const cardClass = isDeny
    ? 'bg-status-deny-bg border-status-deny-text'
    : 'bg-status-review-bg border-status-review-text';
  const textClass = isDeny ? 'text-status-deny-text' : 'text-status-review-text';

  return (
    <div className="bg-white p-0 rounded-lg shadow-sm border border-slate-200 sticky top-10">
      <div className="flex items-center justify-between p-4 border-b border-slate-200">
        <h3 className="text-lg font-semibold text-slate-800">
          Transaction Details
        </h3>
        <button
          onClick={onClose}
          className="text-brand-gray hover:text-slate-800"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      
      <div className="p-4 space-y-4 max-h-[80vh] overflow-y-auto">
        <div className={`p-4 rounded-lg border-l-4 ${cardClass}`}>
          <span
            className={`text-sm font-semibold uppercase ${textClass}`}
          >
            Status: {final_status.replace('_', ' ')}
          </span>
          <p className={`text-3xl font-bold ${textClass}`}>
            {final_score.toFixed(1)}
            <span className="text-lg font-normal"> / 150</span>
          </p>
        </div>
        
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Flagging Reasons
          </h4>
          <div className="space-y-2">
            {reasonsList.map((reason, i) => (
              <div
                key={i}
                className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200"
              >
                {reason}
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Agent Scores
          </h4>
          <div className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200 space-y-1">
            <p><strong>Agent 1 (Threat Intel):</strong> {agent_1_score.toFixed(1)}</p>
            <p><strong>Agent 2 (Behavioral):</strong> {agent_2_score.toFixed(1)}</p>
          </div>
        </div>

        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Transaction Info
          </h4>
          <div className="text-sm space-y-2 font-mono break-words bg-slate-100 p-3 rounded-md border border-slate-200">
            <p>
              <strong>Hash:</strong> {tx_hash}
            </p>
            <p>
              <strong>From:</strong> {from_address}
            </p>
            <p>
              <strong>To:</strong> {to_address}
            </p>
            <p>
              <strong className="font-sans">Value:</strong> {value_eth.toFixed(6)} ETH
            </p>
            <p>
              <strong className="font-sans">Gas:</strong> {gas_price} Gwei
            </p>
            <p>
              <strong className="font-sans">Time:</strong> {new Date(timestamp).toLocaleString()}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TransactionDetail;