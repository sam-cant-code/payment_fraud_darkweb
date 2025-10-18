import { useState, useEffect, useCallback } from 'react';
import {
  getStatus,
  postRunSetup,
  postRunSimulation,
  getFlaggedTransactions,
} from '../services/api';

const useDashboard = () => {
  const [status, setStatus] = useState({
    database_ready: false,
    simulation_run: false,
  });
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState({
    status: true,
    setup: false,
    simulation: false,
    transactions: true,
  });
  const [error, setError] = useState({
    status: null,
    setup: null,
    simulation: null,
    transactions: null,
  });
  const [notification, setNotification] = useState(null); // { message: '', type: 'success' | 'error' }

  // Function to clear notifications
  const clearNotification = () => setNotification(null);
  
  // Function to set notifications (with auto-clear)
  const showNotification = (message, type = 'success') => {
    setNotification({ message, type });
    setTimeout(clearNotification, 5000); // Auto-hide after 5 seconds
  };

  // Fetch initial status
  const fetchStatus = useCallback(async () => {
    setLoading((prev) => ({ ...prev, status: true }));
    setError((prev) => ({ ...prev, status: null }));
    try {
      const response = await getStatus();
      setStatus(response.data);
    } catch (err) {
      setError((prev) => ({ ...prev, status: 'Failed to fetch status' }));
      showNotification('Failed to fetch backend status.', 'error');
    } finally {
      setLoading((prev) => ({ ...prev, status: false }));
    }
  }, []);

  // Fetch flagged transactions
  const fetchTransactions = useCallback(async () => {
    setLoading((prev) => ({ ...prev, transactions: true }));
    setError((prev) => ({ ...prev, transactions: null }));
    try {
      const response = await getFlaggedTransactions();
      setTransactions(response.data);
    } catch (err) {
      // Don't show an error if the file just doesn't exist yet
      if (err.response && err.response.status === 404) {
        setTransactions([]);
        setError((prev) => ({ ...prev, transactions: 'Run simulation to see data.' }));
      } else {
        setError((prev) => ({ ...prev, transactions: 'Failed to fetch transactions' }));
        showNotification('Failed to fetch transactions.', 'error');
      }
    } finally {
      setLoading((prev) => ({ ...prev, transactions: false }));
    }
  }, []);

  // Run setup script
  const runSetup = async () => {
    setLoading((prev) => ({ ...prev, setup: true }));
    setError((prev) => ({ ...prev, setup: null }));
    try {
      const response = await postRunSetup();
      showNotification(response.data.message, 'success');
      await fetchStatus(); // Refresh status after setup
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Setup failed';
      setError((prev) => ({ ...prev, setup: errorMsg }));
      showNotification(errorMsg, 'error');
    } finally {
      setLoading((prev) => ({ ...prev, setup: false }));
    }
  };

  // Run simulation script
  const runSimulation = async () => {
    setLoading((prev) => ({ ...prev, simulation: true }));
    setError((prev) => ({ ...prev, simulation: null }));
    try {
      const response = await postRunSimulation();
      showNotification(response.data.message, 'success');
      await fetchStatus(); // Refresh status
      await fetchTransactions(); // Refresh transactions
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Simulation failed';
      setError((prev) => ({ ...prev, simulation: errorMsg }));
      showNotification(errorMsg, 'error');
    } finally {
      setLoading((prev) => ({ ...prev, simulation: false }));
    }
  };
  
  // Load initial data on mount
  useEffect(() => {
    const loadData = async () => {
      await fetchStatus();
      // Only fetch transactions if simulation has been run
      if (status.simulation_run) {
        await fetchTransactions();
      } else {
         setLoading((prev) => ({ ...prev, transactions: false }));
      }
    };
    loadData();
  }, [status.simulation_run]); // Re-run if simulation_run status changes

  return {
    status,
    transactions,
    loading,
    error,
    notification,
    runSetup,
    runSimulation,
    fetchTransactions, // Expose this for a manual refresh button
    clearNotification,
  };
};

export default useDashboard;