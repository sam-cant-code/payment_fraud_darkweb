import React from 'react';
import { X, AlertTriangle, Check, ShieldOff, FileWarning, Loader2 } from 'lucide-react'; // Added icons

// Added onSubmitReview prop and loading state (passed down from parent)
const TransactionDetail = ({ transaction, onClose, onSubmitReview, isLoadingReview }) => {
  if (!transaction) {
     return (
      <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200 h-full flex flex-col items-center justify-center text-center sticky top-6"> {/* Made sticky */}
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
    agent_3_score, // Keep displaying if available, even if score is 0
  } = transaction;

  const reasonsList = reasons ? reasons.split(' | ') : []; // Initialize as empty array
  const isApproved = final_status === 'APPROVE';
  const isDeny = final_status === 'DENY';
  const isReview = final_status === 'FLAG_FOR_REVIEW';

  const cardClass = isApproved
    ? 'bg-status-approve-bg border-status-approve-text'
    : isDeny
    ? 'bg-status-deny-bg border-status-deny-text'
    : isReview // Default to review style if not approved or deny
    ? 'bg-status-review-bg border-status-review-text'
    : 'bg-slate-100 border-slate-300'; // Fallback style
  const textClass = isApproved
    ? 'text-status-approve-text'
    : isDeny
    ? 'text-status-deny-text'
    : isReview
    ? 'text-status-review-text'
    : 'text-slate-700'; // Fallback style

  const handleReviewClick = (newStatus) => {
    if (onSubmitReview && !isLoadingReview) {
      onSubmitReview(tx_hash, newStatus);
    }
  };

  // Determine the max score based on whether Agent 3 contributes (use 130 if neutralized, 150 otherwise)
  // Assuming neutralization:
  const MAX_POSSIBLE_SCORE = 130;

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
          aria-label="Close transaction details"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Make content scrollable */}
      <div className="p-4 space-y-4 max-h-[calc(100vh-12rem)] overflow-y-auto"> {/* Adjust max-h as needed */}
        {/* Status and Score */}
        <div className={`p-4 rounded-lg border-l-4 ${cardClass}`}>
           <div className="flex items-center gap-2">
             {/* Show relevant icon based on status */}
             {isApproved && <Check className={`w-5 h-5 ${textClass}`} />}
             {isReview && <FileWarning className={`w-5 h-5 ${textClass}`} />}
             {isDeny && <ShieldOff className={`w-5 h-5 ${textClass}`} />}
            <span className={`text-sm font-semibold uppercase ${textClass}`}>
              Status: {final_status.replace(/_/g, ' ')}
            </span>
           </div>
          <p className={`text-3xl font-bold ${textClass} mt-1`}>
            {final_score.toFixed(1)}
            <span className="text-lg font-normal"> / {MAX_POSSIBLE_SCORE}</span> {/* Use dynamic Max Score */}
          </p>
        </div>

        {/* Analyst Review Actions */}
        {/* Show only if Deny or Review, Approved txns don't need review actions */}
        {(isDeny || isReview) && (
             <div>
                <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
                    Analyst Review
                </h4>
                <div className="flex gap-2">
                    {/* Approve Button (Visible for DENY and REVIEW) */}
                    <button
                        onClick={() => handleReviewClick('APPROVE')}
                        disabled={isLoadingReview}
                        className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-approve-bg text-status-approve-text hover:brightness-95 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <Check className="w-4 h-4" />} Approve
                    </button>

                    {/* Deny Button (Only visible if current status is REVIEW) */}
                     {isReview && (
                         <button
                            onClick={() => handleReviewClick('DENY')}
                            disabled={isLoadingReview}
                             className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-deny-bg text-status-deny-text hover:brightness-95 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                             {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <ShieldOff className="w-4 h-4" />} Deny
                        </button>
                    )}

                     {/* Re-Flag Button (Only visible if current status is DENY) */}
                     {isDeny && (
                         <button
                            onClick={() => handleReviewClick('FLAG_FOR_REVIEW')}
                            disabled={isLoadingReview}
                             className="flex-1 flex items-center justify-center gap-1 px-3 py-2 rounded-md bg-status-review-bg text-status-review-text hover:brightness-95 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                             {isLoadingReview ? <Loader2 className="w-4 h-4 animate-spin"/> : <FileWarning className="w-4 h-4" />} Re-Flag
                        </button>
                    )}
                </div>
            </div>
        )}

        {/* Reasons / Analysis Summary */}
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Analysis Summary
          </h4>
          <div className="space-y-2">
            {/* Show specific message for clean Approved transactions */}
            {isApproved && (reasonsList.length === 0 || (reasonsList.length === 1 && reasonsList[0].startsWith("No specific"))) ? (
              <div className="text-sm bg-status-approve-bg text-status-approve-text p-3 rounded-md border border-status-approve-text font-medium">
                ✅ Approved: Low risk score detected. No significant flags raised by analysis agents.
              </div>
            ) : (
              // Otherwise, display the reasons provided
              reasonsList.map((reason, i) => (
                <div key={i} className={`text-sm p-3 rounded-md border ${
                    // Use background/border based on severity inferred from keywords
                    reason.includes('Dark Web') || reason.includes('DENY') || reason.includes('Very high-value')
                    ? 'bg-status-deny-bg border-status-deny-text text-status-deny-text'
                    : reason.includes('anomaly') || reason.includes('higher than wallet avg') || reason.includes('High-value') || reason.includes('linked to known threat')
                    ? 'bg-status-review-bg border-status-review-text text-status-review-text'
                    : 'bg-slate-100 border-slate-200 text-slate-700' // Default style
                }`}>
                  {reason}
                </div>
              ))
            )}
             {/* Handle case where reasons might be unexpectedly empty for non-approved */}
             {reasonsList.length === 0 && !isApproved && (
                 <div className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200 text-slate-500 italic">
                    No specific reasons provided by agents for this status.
                 </div>
             )}
          </div>
        </div>

        {/* Agent Scores */}
        <div>
          <h4 className="text-xs uppercase text-brand-gray font-semibold mb-2">
            Agent Scores
          </h4>
          {/* Display scores - handle potentially missing scores gracefully */}
          <div className="text-sm bg-slate-100 p-3 rounded-md border border-slate-200 space-y-1">
            <p><strong>Agent 1 (Threat Intel):</strong> {(agent_1_score ?? 0).toFixed(1)}</p>
            <p><strong>Agent 2 (Behavioral):</strong> {(agent_2_score ?? 0).toFixed(1)}</p>
            <p><strong>Agent 3 (Network):</strong> {(agent_3_score ?? 0).toFixed(1)}</p> {/* Display Agent 3 score (even if 0) */}
          </div>
        </div>

        {/* Transaction Info */}
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
              <strong className="font-sans">Gas:</strong> {gas_price.toFixed(2)} Gwei
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