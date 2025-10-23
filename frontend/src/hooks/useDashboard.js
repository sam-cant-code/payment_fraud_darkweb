// frontend/src/hooks/useDashboard.js
import { useState, useEffect, useCallback, useRef } from 'react';
import {
  getStatus,
  postRunSetup,
  getFlaggedTransactions, // Renamed but still fetches all initial for store
  getWalletProfile,
  postReview,
  getThreats,
  postThreat,
  deleteThreat,
  socket
} from '../services/api';
import { useTransactionStore } from '../store/transactionStore'; // Import the store hook

// const MAX_TRANSACTIONS = 200; // Limit is now handled in the store

const useDashboard = () => {
  // --- Zustand State Access ---
  // Get state and actions from the store
  const {
    setTransactions,
    addTransaction,
    updateTransaction,
    // Note: We don't need 'transactions' or 'filteredTransactions' directly in this hook anymore
  } = useTransactionStore.getState(); // Get actions synchronously if needed outside components

  // --- Local State (for things not in Zustand) ---
  const [status, setStatus] = useState({
    database_ready: false,
    simulation_running: false,
    threat_list_available: false,
    has_flagged_data: false,
    websocket_connected: false,
    listener_active: false,
  });
  const [loading, setLoading] = useState({
    status: true,
    setup: false,
    transactions: false, // For initial load flag
    review: false,
    threats: true,
    walletProfile: false,
  });
  const [error, setError] = useState({ status: null, setup: null, transactions: null, review: null, threats: null, walletProfile: null });
  const [notification, setNotification] = useState(null);
  const [threatList, setThreatList] = useState([]);
  const [selectedWalletProfile, setSelectedWalletProfile] = useState(null);

  const isInitialLoadDone = useRef(false);

  // --- Notifications (keep as is) ---
  const clearNotification = useCallback(() => setNotification(null), []);
  const showNotification = useCallback((message, type = 'success', duration = 5000) => {
      setNotification({ message, type });
      const timerId = setTimeout(clearNotification, duration);
      return () => clearTimeout(timerId);
  }, [clearNotification]);


  // --- Data Fetching Callbacks (fetchStatus, fetchThreatList remain mostly the same) ---
  const fetchStatus = useCallback(async () => {
    if (!isInitialLoadDone.current) {
        setLoading((prev) => ({ ...prev, status: true }));
    }
    setError((prev) => ({ ...prev, status: null }));
    try {
      const response = await getStatus();
      const backendStatus = response.data;
      setStatus(prevStatus => ({
          ...prevStatus,
          database_ready: backendStatus.database_ready ?? false,
          simulation_running: backendStatus.simulation_running ?? false,
          listener_active: backendStatus.listener_active ?? false,
          threat_list_available: backendStatus.threat_list_available ?? false,
          has_flagged_data: backendStatus.has_flagged_data ?? false,
      }));
    } catch (err) {
      setError((prev) => ({ ...prev, status: 'Failed to fetch status' }));
      showNotification('Failed to fetch backend status.', 'error');
      console.error("Status fetch error:", err);
    } finally {
       if (!isInitialLoadDone.current) {
         setLoading((prev) => ({ ...prev, status: false }));
       }
    }
  }, [showNotification]);

  // Fetches initial transactions and populates the Zustand store
  const fetchInitialTransactions = useCallback(async () => {
    setLoading((prev) => ({ ...prev, transactions: true })); // Use transactions flag for initial load
    setError((prev) => ({ ...prev, transactions: null }));
    try {
      const response = await getFlaggedTransactions();
      const sortedTransactions = response.data.sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
      );
      // Use the store action to set transactions
      setTransactions(sortedTransactions); // Limit is handled within the store action now
      console.log(`Loaded initial ${sortedTransactions.length} transactions into store.`);
    } catch (err) {
       if (err.response && err.response.status === 404) {
         setTransactions([]); // Set empty in store
         console.log("Initial transaction file not found or empty (API returned 404).");
      } else {
        setError((prev) => ({ ...prev, transactions: 'Failed to fetch initial transactions' }));
        showNotification('Failed to fetch initial transactions.', 'error');
        console.error("Initial transaction fetch error:", err);
      }
    } finally {
      setLoading((prev) => ({ ...prev, transactions: false }));
    }
     // Include setTransactions from the store in dependencies if ESLint complains, though it should be stable
  }, [showNotification, setTransactions]);

  const fetchThreatList = useCallback(async () => {
     if (!isInitialLoadDone.current) {
        setLoading((prev) => ({ ...prev, threats: true }));
     }
    setError((prev) => ({ ...prev, threats: null }));
    try {
      const response = await getThreats();
      setThreatList(response.data);
    } catch (err) {
      setError((prev) => ({ ...prev, threats: 'Failed to fetch threat list' }));
      showNotification('Failed to load threat list.', 'error');
      console.error("Threat list fetch error:", err);
    } finally {
      if (!isInitialLoadDone.current) {
         setLoading((prev) => ({ ...prev, threats: false }));
      }
    }
  }, [showNotification]);


  // --- Action Callbacks (runSetup, submitTxReview, add/remove threats, fetchWalletDetails remain mostly the same) ---
   const runSetup = useCallback(async () => {
    setLoading((prev) => ({ ...prev, setup: true }));
    setError((prev) => ({ ...prev, setup: null }));
    try {
      const response = await postRunSetup();
      showNotification(response.data.message, 'success');
      setTransactions([]); // Clear transactions in store after setup
      await fetchStatus();
      await fetchThreatList();
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Setup failed';
      setError((prev) => ({ ...prev, setup: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Setup error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, setup: false }));
    }
  }, [fetchStatus, showNotification, fetchThreatList, setTransactions]); // Added setTransactions

  const submitTxReview = useCallback(async (txHash, newStatus) => {
    setLoading((prev) => ({ ...prev, review: true }));
    setError((prev) => ({ ...prev, review: null }));
    try {
      const response = await postReview(txHash, newStatus);
      showNotification(response.data.message, 'success');
      // Rely on 'transaction_update' event from websocket
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Review submission failed';
      setError((prev) => ({ ...prev, review: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Review error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, review: false }));
    }
  }, [showNotification]);

 const addWalletToThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true }));
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await postThreat(walletAddress);
        showNotification(`Attempting to add ${walletAddress.substring(0,10)}...`, 'info', 2000);
    } catch (err) {
       const errorMsg = err.response?.data?.error || 'Failed to add wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Add threat error:", err);
       setLoading((prev) => ({ ...prev, threats: false }));
    }
}, [showNotification]);

const removeWalletFromThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true }));
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await deleteThreat(walletAddress);
        showNotification(`Attempting to remove ${walletAddress.substring(0,10)}...`, 'info', 2000);
    } catch (err) {
       const errorMsg = err.response?.data?.error || 'Failed to remove wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Remove threat error:", err);
       setLoading((prev) => ({ ...prev, threats: false }));
    }
}, [showNotification]);

  const fetchWalletDetails = useCallback(async (address) => {
    if (!address) return;
    setLoading(prev => ({ ...prev, walletProfile: true }));
    setError(prev => ({ ...prev, walletProfile: null }));
    setSelectedWalletProfile(null);
    try {
      const response = await getWalletProfile(address);
      setSelectedWalletProfile(response.data);
      console.log("Fetched profile for:", address, response.data);
    } catch (err) {
      const errorMsg = err.response?.status === 404
        ? `No profile found for wallet ${address.substring(0, 10)}...`
        : 'Failed to fetch wallet profile.';
      setError(prev => ({ ...prev, walletProfile: errorMsg }));
      showNotification(errorMsg, err.response?.status === 404 ? 'info' : 'error');
      console.error("Wallet profile fetch error:", err);
    } finally {
      setLoading(prev => ({ ...prev, walletProfile: false }));
    }
  }, [showNotification]);


  // --- Effects ---

  // Effect for Initial Load and WebSocket Management
  useEffect(() => {
    const performInitialLoad = async () => {
        console.log("Performing initial data load...");
        setLoading(prev => ({ ...prev, status: true, threats: true, transactions: true })); // Set initial transaction loading true
        await fetchStatus();
        await fetchThreatList();
        await fetchInitialTransactions(); // Fetch initial transactions into store
        isInitialLoadDone.current = true;
        setLoading(prev => ({ ...prev, status: false, threats: false, transactions: false }));
        console.log("Initial data load complete.");
    };

    performInitialLoad();

    // --- WebSocket Event Handlers ---
    const onConnect = () => {
      console.log('Socket connected');
      setStatus(prev => ({ ...prev, websocket_connected: true }));
      showNotification('Real-time connection established.', 'success', 3000);
      fetchStatus();
    };

    const onDisconnect = (reason) => {
      console.log('Socket disconnected:', reason);
      setStatus(prev => ({ ...prev, websocket_connected: false, listener_active: false }));
      showNotification('Real-time connection lost.', 'error');
    };

    // Use the store action to add new transactions
    const onNewTransaction = (newTransaction) => {
      console.log('Received new scanned transaction:', newTransaction);
      addTransaction(newTransaction); // Use store action
       if (newTransaction.final_status !== 'APPROVE') {
           const messageType = newTransaction.final_status === 'DENY' ? 'error' : 'warning';
           showNotification(`Tx ${newTransaction.tx_hash.substring(0,10)}... ${newTransaction.final_status.replace('_',' ')}.`, messageType, 4000);
       }
    };

    // Use the store action to update transactions
    const onTransactionUpdate = (updatedTransaction) => {
       console.log('Received transaction update:', updatedTransaction);
       updateTransaction(updatedTransaction); // Use store action
       showNotification(`Tx ${updatedTransaction.tx_hash.substring(0,10)}... review updated.`, 'info', 3000);
    };

    const onThreatListUpdate = (updatedList) => {
        console.log('Received threat list update');
        setThreatList(updatedList);
        setLoading((prev) => ({ ...prev, threats: false }));
        showNotification('Threat list updated.', 'info', 3000);
    };

    // --- Connect and Register Listeners ---
    if (!socket.connected) {
      console.log("Attempting to connect socket...");
      socket.connect();
    }
    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('new_scanned_transaction', onNewTransaction);
    socket.on('transaction_update', onTransactionUpdate);
    socket.on('threat_list_updated', onThreatListUpdate);

    // --- Cleanup Function ---
    return () => {
      console.log("Cleaning up socket listeners...");
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('new_scanned_transaction', onNewTransaction);
      socket.off('transaction_update', onTransactionUpdate);
      socket.off('threat_list_updated', onThreatListUpdate);
    };
     // Add store actions to dependency array if needed, they should be stable
  }, [showNotification, fetchStatus, fetchThreatList, fetchInitialTransactions, addTransaction, updateTransaction]);


  return {
    // Return local state needed by UI
    status,
    loading,
    error,
    notification,
    threatList,
    selectedWalletProfile,
    // Return actions needed by UI
    runSetup,
    clearNotification,
    fetchThreatList, // Keep for potential manual refresh in UI
    addWalletToThreats,
    removeWalletFromThreats,
    submitTxReview,
    fetchWalletDetails,
    // Note: Don't need to return transactions state/setters from here anymore
  };
};

export default useDashboard;