// frontend/src/pages/SimulationPage.jsx
import React, { useState } from 'react';
import { postSimulateTransaction } from '../services/api';
import { Send, Loader2, AlertTriangle, CheckCircle, Zap } from 'lucide-react';

const SimulationPage = () => {
  const [formData, setFormData] = useState({
    from_address: '',
    to_address: '',
    value_eth: '',
    gas_price: '',
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState({ type: '', message: '' }); // 'success' or 'error'

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setSubmitStatus({ type: '', message: '' });

    try {
      const txData = {
        from_address: formData.from_address,
        to_address: formData.to_address,
        value_eth: parseFloat(formData.value_eth),
        gas_price: formData.gas_price ? parseFloat(formData.gas_price) : undefined, // Send undefined if empty
      };

      if (!txData.from_address || !txData.to_address || isNaN(txData.value_eth)) {
        throw new Error('From, To, and Value (ETH) are required.');
      }

      const response = await postSimulateTransaction(txData);
      setSubmitStatus({
        type: 'success',
        message: `Transaction simulated! Status: ${response.data.transaction.final_status}, Score: ${response.data.transaction.final_score.toFixed(0)}. Check the dashboard.`,
      });
      // Clear form on success
      setFormData({ from_address: '', to_address: '', value_eth: '', gas_price: '' });

    } catch (err) {
      const errorMsg = err.response?.data?.error || err.message || 'Failed to submit simulation.';
      setSubmitStatus({ type: 'error', message: errorMsg });
    } finally {
      setIsSubmitting(false);
    }
  };

  // --- Example Data ---
  // A known scam wallet from your dark_web_wallets.txt
  const scamWallet = '0x0d39e6277a069095033c3a0e1c36bd2a0a2f768b';
  // A 'normal' looking wallet
  const normalWallet = '0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B'; // (Vitalik's)
  // A brand new, random wallet
  const newWallet = '0x' + [...Array(40)].map(() => Math.floor(Math.random() * 16).toString(16)).join('');


  const loadExample = (exampleType) => {
    setSubmitStatus({ type: '', message: '' });
    switch (exampleType) {
      case 'scam':
        setFormData({
          from_address: normalWallet,
          to_address: scamWallet,
          value_eth: '0.5',
          gas_price: '50',
        });
        break;
      case 'high_risk_new':
        setFormData({
          from_address: '0x1234567890123456789012345678901234567890', // Another new wallet
          to_address: newWallet,
          value_eth: '80.0', // High value
          gas_price: '20',
        });
        break;
      case 'normal':
        setFormData({
          from_address: normalWallet,
          to_address: '0xdAC17F958D2ee523a2206206994597C13D831ec7', // (USDT contract)
          value_eth: '1.2',
          gas_price: '45',
        });
        break;
      default:
        setFormData({ from_address: '', to_address: '', value_eth: '', gas_price: '' });
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-sm border border-slate-200">
        <h2 className="text-2xl font-semibold text-slate-800 mb-4">
          Simulate a Transaction
        </h2>
        <p className="text-sm text-slate-600 mb-6">
          Submit custom transaction data to be processed by the AI pipeline. The
          result will appear on the dashboard in real-time.
        </p>

        {/* --- Quick Load Examples --- */}
        <div className="mb-6">
          <h3 className="text-lg font-medium text-slate-700 mb-3">
            Load Example
          </h3>
          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => loadExample('scam')}
              className="flex items-center gap-2 text-sm px-4 py-2 bg-status-deny-bg text-status-deny-text rounded-md hover:bg-status-deny-bg/80"
            >
              <AlertTriangle className="w-4 h-4" />
              To Known Scammer
            </button>
            <button
              onClick={() => loadExample('high_risk_new')}
              className="flex items-center gap-2 text-sm px-4 py-2 bg-status-review-bg text-status-review-text rounded-md hover:bg-status-review-bg/80"
            >
              <Zap className="w-4 h-4" />
              High Value to New Wallet
            </button>
            <button
              onClick={() => loadExample('normal')}
              className="flex items-center gap-2 text-sm px-4 py-2 bg-status-approve-bg text-status-approve-text rounded-md hover:bg-status-approve-bg/80"
            >
              <CheckCircle className="w-4 h-4" />
              Normal Transaction
            </button>
          </div>
        </div>

        {/* --- Manual Form --- */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="from_address"
                className="block text-sm font-medium text-slate-700 mb-1"
              >
                From Address (Sender) <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="from_address"
                id="from_address"
                value={formData.from_address}
                onChange={handleInputChange}
                required
                className="w-full p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue font-mono"
                placeholder="0x..."
              />
            </div>
            <div>
              <label
                htmlFor="to_address"
                className="block text-sm font-medium text-slate-700 mb-1"
              >
                To Address (Receiver) <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                name="to_address"
                id="to_address"
                value={formData.to_address}
                onChange={handleInputChange}
                required
                className="w-full p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue font-mono"
                placeholder="0x..."
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="value_eth"
                className="block text-sm font-medium text-slate-700 mb-1"
              >
                Value (ETH) <span className="text-red-500">*</span>
              </label>
              <input
                type="number"
                name="value_eth"
                id="value_eth"
                value={formData.value_eth}
                onChange={handleInputChange}
                required
                step="any"
                min="0"
                className="w-full p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                placeholder="e.g., 0.5"
              />
            </div>
            <div>
              <label
                htmlFor="gas_price"
                className="block text-sm font-medium text-slate-700 mb-1"
              >
                Gas Price (Gwei) <span className="text-slate-400">(Optional)</span>
              </label>
              <input
                type="number"
                name="gas_price"
                id="gas_price"
                value={formData.gas_price}
                onChange={handleInputChange}
                step="any"
                min="0"
                className="w-full p-2 border border-slate-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-brand-blue"
                placeholder="e.g., 50 (if blank, uses random)"
              />
            </div>
          </div>

          <div className="flex items-center justify-between pt-4">
            <button
              type="submit"
              disabled={isSubmitting}
              className="flex items-center justify-center gap-2 px-6 py-2.5 bg-brand-blue text-white font-medium rounded-md shadow-sm hover:bg-brand-blue/90 focus:outline-none focus:ring-2 focus:ring-brand-blue focus:ring-offset-2 disabled:opacity-50"
            >
              {isSubmitting ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
              <span>{isSubmitting ? 'Submitting...' : 'Submit Simulation'}</span>
            </button>

            {/* --- Status Message --- */}
            {submitStatus.message && (
              <div
                className={`text-sm ${
                  submitStatus.type === 'success' ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {submitStatus.message}
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default SimulationPage;