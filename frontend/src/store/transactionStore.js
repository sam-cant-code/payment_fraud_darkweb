// frontend/src/store/transactionStore.js
import { create } from 'zustand';

// Helper function for filtering
const filterTransactions = (transactions, filters) => {
  return transactions.filter(tx => {
    // --- Text Filters (case-insensitive partial match) ---
    if (filters.sender && !tx.from_address.toLowerCase().includes(filters.sender.toLowerCase())) {
      return false;
    }
    if (filters.receiver && !tx.to_address.toLowerCase().includes(filters.receiver.toLowerCase())) {
      return false;
    }
    if (filters.txHash && !tx.tx_hash.toLowerCase().includes(filters.txHash.toLowerCase())) {
      return false;
    }

    // --- Range Filters ---
    // Value (ETH)
    if (filters.minValue !== null && tx.value_eth < filters.minValue) {
      return false;
    }
    if (filters.maxValue !== null && tx.value_eth > filters.maxValue) {
      return false;
    }

    // Risk Score
    if (filters.minRisk !== null && tx.final_score < filters.minRisk) {
      return false;
    }
    if (filters.maxRisk !== null && tx.final_score > filters.maxRisk) {
      return false;
    }

    // If none of the filters excluded the transaction, include it
    return true;
  });
};


export const useTransactionStore = create((set, get) => ({
  // --- State ---
  allTransactions: [], // Holds all fetched/received transactions
  filteredTransactions: [], // Holds the transactions after filtering
  filters: {
    sender: '',
    receiver: '',
    txHash: '',
    minValue: null,
    maxValue: null,
    minRisk: null,
    maxRisk: null,
  },

  // --- Actions ---
  setTransactions: (transactions) => set(state => {
    const newState = { allTransactions: transactions };
    // Re-apply filters whenever the base list changes
    newState.filteredTransactions = filterTransactions(transactions, state.filters);
    return newState;
  }),

  addTransaction: (newTransaction) => set(state => {
    // Prevent duplicates and add to the start
    if (state.allTransactions.some(tx => tx.tx_hash === newTransaction.tx_hash)) {
      return {}; // No change if duplicate
    }
    const updatedAllTransactions = [newTransaction, ...state.allTransactions].slice(0, 200); // Limit size
    const updatedFilteredTransactions = filterTransactions(updatedAllTransactions, state.filters);
    return {
      allTransactions: updatedAllTransactions,
      filteredTransactions: updatedFilteredTransactions,
    };
  }),

  updateTransaction: (updatedTransaction) => set(state => {
    const updatedAllTransactions = state.allTransactions.map(tx =>
      tx.tx_hash === updatedTransaction.tx_hash ? updatedTransaction : tx
    );
    const updatedFilteredTransactions = filterTransactions(updatedAllTransactions, state.filters);
    return {
      allTransactions: updatedAllTransactions,
      filteredTransactions: updatedFilteredTransactions,
    };
  }),

  setFilter: (filterName, value) => set(state => {
    const newFilters = { ...state.filters, [filterName]: value };
    const newFilteredTransactions = filterTransactions(state.allTransactions, newFilters);
    return {
      filters: newFilters,
      filteredTransactions: newFilteredTransactions,
    };
  }),

  // Optional: Action to clear all filters
  clearFilters: () => set(state => {
     const defaultFilters = {
       sender: '', receiver: '', txHash: '',
       minValue: null, maxValue: null,
       minRisk: null, maxRisk: null,
     };
     const newFilteredTransactions = filterTransactions(state.allTransactions, defaultFilters);
     return {
        filters: defaultFilters,
        filteredTransactions: newFilteredTransactions,
     };
  }),
}));