import React from 'react';
import { X, AlertTriangle, Check, ShieldOff, FileWarning, Loader2 } from 'lucide-react'; // Added icons

// Added onSubmitReview prop and loading state
const TransactionDetail = ({ transaction, onClose, onSubmitReview, isLoadingReview }) => {
  if (!transaction) {
    // ... (placeholder remains the same) ...
     return (
      <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 h-full flex flex-col items-center justify-center text-center sticky top-10">
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
    agent_3_score, // Added agent 3 score display
  } = transaction;

  const reasonsList = reasons ? reasons.split(' | ') : ["No specific reasons provided."];
  const isDeny = final_status === 'DENY';
  const isReview = final_status === 'FLAG_FOR_REVIEW';

  const cardClass = isDeny
    ? 'bg-status-deny-bg border-status-deny-text'
    : isReview
    ? 'bg-status-review-bg border-status-review-text'
    : 'bg-status-approve-bg border-status-approve-text'; // Added approve style
  const textClass = isDeny
    ? 'text-status-deny-text'
    : isReview
    ? 'text-status-review-text'
    : 'text-status-approve-text';

  const handleReviewClick = (newStatus) => {
    if (onSubmitReview && !isLoadingReview) {
      onSubmitReview(tx_hash, newStatus);
    }
  };

  return (
    // Make it sticky within its column
    <div className="bg-white rounded-lg shadow-sm border border-slate-200 sticky top-6">
      <div className="flex items-center justify-between p-4 border-b border-slate-200">
        <h3 className="text-lg font-semibold text-slate-800 truncate pr-2">
          Tx: {tx_hash.substring(0, 12)}...
        </h3>
        <button
          onClick={onClose}
          className="text-brand-gray hover:text-slate-800 flex-shrink-0"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Make content scrollable */}
      <div className="p-4 space-y-4 max-h-[calc(100vh-12rem)] overflow-y-auto"> {/* Adjust max-h as needed */}
        {/* Status and Score */}
        <div className={`p-4 rounded-lg border-l-4 ${cardClass}`}>
          <span className={`text-sm font-semibold uppercase ${textClass}`}>
            Status: {final_status.replace(/_/g, ' ')}
          </span>
          <p className={`text-3xl font-bold ${textClass}`}>
            {final_score.toFixed(1)}
            <span className="text-lg font-normal"> / 150</span>
          </p>
        </div>

        {/* Analyst Review Actions */}
        {(isDeny || isReview) && ( // Show only if Deny or Review
             <div>
                <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
                    Analyst Review
                </h4>
                <div className="flex gap-2">
                    <button
                        onClick={() => handleReviewClick('APPROVE')}
                        disabled={isLoadingReview}
                        className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-approve-bg text-status-approve-text hover:brightness-95 text-sm font-medium disabled:opacity-50"
                    >
                        {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <Check className="w-4 h-4" />} Approve
                    </button>
                    {isReview && ( // Show Deny button only if current status is Review
                         <button
                            onClick={() => handleReviewClick('DENY')}
                            disabled={isLoadingReview}
                             className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-deny-bg text-status-deny-text hover:brightness-95 text-sm font-medium disabled:opacity-50"
                        >
                             {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <ShieldOff className="w-4 h-4" />} Deny
                        </button>
                    )}
                     {isDeny && ( // Show Re-Flag button only if current status is Deny
                         <button
                            onClick={() => handleReviewClick('FLAG_FOR_REVIEW')}
                            disabled={isLoadingReview}
                             className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-review-bg text-status-review-text hover:brightness-95 text-sm font-medium disabled:opacity-50"
                        >
                             {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <FileWarning className="w-4 h-4" />} Re-Flag
                        </button>
                    )}
                </div>
            </div>
        )}


        {/* Reasons */}
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Flagging Reasons
          </h4>
          <div className="space-y-2">
            {reasonsList.map((reason, i) => (
              <div key={i} className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200">
                {reason}
              </div>
            ))}
            {reasonsList.length === 1 && reasonsList[0].startsWith("No specific") && (
                 <p className="text-sm text-slate-500 italic px-1">Transaction was approved by agents.</p>
            )}
          </div>
        </div>

        {/* Agent Scores */}
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Agent Scores
          </h4>
          <div className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200 space-y-1">
            <p><strong>Agent 1 (Threat Intel):</strong> {(agent_1_score ?? 0).toFixed(1)}</p>
            <p><strong>Agent 2 (Behavioral):</strong> {(agent_2_score ?? 0).toFixed(1)}</p>
            <p><strong>Agent 3 (Network):</strong> {(agent_3_score ?? 0).toFixed(1)}</p> {/* Display Agent 3 */}
          </div>
        </div>

        {/* Transaction Info */}
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Transaction Info
          </h4>
          <div className="text-sm space-y-2 font-mono break-words bg-slate-100 p-3 rounded-md border border-slate-200">
             {/* ... (keep tx_hash, from, to etc. as before) ... */}
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