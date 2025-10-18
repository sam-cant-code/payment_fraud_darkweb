import { useState, useEffect, useCallback, useRef } from 'react'; // Added useRef
import {
  getStatus,
  postRunSetup,
  // postRunSimulation, // Removed
  getFlaggedTransactions, // Still needed for initial load
  getWalletProfile,
  postReview,
  getThreats,
  postThreat,
  deleteThreat,
  socket // Import the socket instance
} from '../services/api';

const MAX_TRANSACTIONS = 200; // Limit the number of transactions kept in state

const useDashboard = () => {
  // --- State ---
  const [status, setStatus] = useState({
    database_ready: false,
    simulation_running: false, // Changed from simulation_run
    threat_list_available: false,
    has_flagged_data: false, // Added from backend status
    websocket_connected: false, // Added connection status
  });
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState({
    status: true,
    setup: false,
    // simulation: false, // Removed
    transactions: true, // For initial load
    review: false,
    threats: true, // Load threats initially
    walletProfile: false,
  });
  const [error, setError] = useState({ /* ... as before ... */ });
  const [notification, setNotification] = useState(null);
  const [threatList, setThreatList] = useState([]);
  const [selectedWalletProfile, setSelectedWalletProfile] = useState(null);

  // Ref to track if initial load is done
  const isInitialLoadDone = useRef(false);

  // --- Notifications ---
  const clearNotification = useCallback(() => setNotification(null), []);
  const showNotification = useCallback((message, type = 'success', duration = 5000) => {
      setNotification({ message, type });
      const timerId = setTimeout(clearNotification, duration);
      return () => clearTimeout(timerId); // Cleanup timer
  }, [clearNotification]);


  // --- Data Fetching Callbacks ---
  const fetchStatus = useCallback(async () => {
    // Only set loading true if it's the very first fetch
    if (!isInitialLoadDone.current) {
        setLoading((prev) => ({ ...prev, status: true }));
    }
    setError((prev) => ({ ...prev, status: null }));
    try {
      const response = await getStatus();
      setStatus(prevStatus => ({
          ...prevStatus, // Keep websocket_connected status
          ...response.data // Update with API response
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

  // Fetches the *initial* list of transactions from the CSV/DB log
  const fetchInitialTransactions = useCallback(async () => {
    setLoading((prev) => ({ ...prev, transactions: true }));
    setError((prev) => ({ ...prev, transactions: null }));
    try {
      const response = await getFlaggedTransactions();
      // Sort by timestamp descending for initial view (newest first)
      const sortedTransactions = response.data.sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
      );
      setTransactions(sortedTransactions.slice(0, MAX_TRANSACTIONS)); // Apply limit
      console.log(`Loaded initial ${sortedTransactions.length} transactions.`);
    } catch (err) {
      // Handle 404 as empty list, other errors as errors
       if (err.response && err.response.status === 404) {
        setTransactions([]);
        console.log("Initial transaction file not found or empty.");
      } else {
        setError((prev) => ({ ...prev, transactions: 'Failed to fetch initial transactions' }));
        showNotification('Failed to fetch initial transactions.', 'error');
        console.error("Initial transaction fetch error:", err);
      }
    } finally {
      setLoading((prev) => ({ ...prev, transactions: false }));
    }
  }, [showNotification]);

  const fetchThreatList = useCallback(async () => {
     if (!isInitialLoadDone.current) { // Only show loading on initial load
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

  // --- Action Callbacks ---
  const runSetup = useCallback(async () => {
    setLoading((prev) => ({ ...prev, setup: true }));
    setError((prev) => ({ ...prev, setup: null }));
    try {
      const response = await postRunSetup();
      showNotification(response.data.message, 'success');
      await fetchStatus(); // Refresh status
    } catch (err) { // Error handling as before
      const errorMsg = err.response?.data?.error || 'Setup failed';
      setError((prev) => ({ ...prev, setup: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Setup error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, setup: false }));
    }
  }, [fetchStatus, showNotification]);

  // runSimulation is no longer needed as a direct user action

  const submitTxReview = useCallback(async (txHash, newStatus) => {
    // (Keep implementation as before - backend now handles the socket emit)
    setLoading((prev) => ({ ...prev, review: true }));
    setError((prev) => ({ ...prev, review: null }));
    try {
      const response = await postReview(txHash, newStatus);
      showNotification(response.data.message, 'success');
      // OPTIONAL: Immediately update local state for faster UI feedback,
      // but rely on the websocket 'transaction_update' for the source of truth
      // setTransactions(prev => prev.map(tx =>
      //   tx.tx_hash === txHash ? { ...tx, final_status: newStatus } : tx
      // ));
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Review submission failed';
      setError((prev) => ({ ...prev, review: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Review error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, review: false }));
    }
  }, [showNotification]);

  // addWalletToThreats & removeWalletFromThreats can remain mostly the same,
  // but rely on 'threat_list_updated' event if implemented, or call fetchThreatList
 const addWalletToThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true }));
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await postThreat(walletAddress);
        // Backend now emits 'threat_list_updated', no local update needed here
        // If not using emit, uncomment: await fetchThreatList();
    } catch (err) { // Error handling as before...
       const errorMsg = err.response?.data?.error || 'Failed to add wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Add threat error:", err);
    } finally {
       setLoading((prev) => ({ ...prev, threats: false })); // Stop loading anyway
    }
}, [showNotification]); // Removed fetchThreatList

const removeWalletFromThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true }));
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await deleteThreat(walletAddress);
         // Backend now emits 'threat_list_updated', no local update needed here
         // If not using emit, uncomment: await fetchThreatList();
    } catch (err) { // Error handling as before...
       const errorMsg = err.response?.data?.error || 'Failed to remove wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Remove threat error:", err);
    } finally {
       setLoading((prev) => ({ ...prev, threats: false }));
    }
}, [showNotification]); // Removed fetchThreatList

  const fetchWalletDetails = useCallback(async (address) => { /* ... as before ... */ }, [showNotification]);

  // --- Effects ---

  // Effect for Initial Load and WebSocket Management
  useEffect(() => {
    const performInitialLoad = async () => {
        console.log("Performing initial data load...");
        await fetchStatus();
        await fetchThreatList();
        // Fetch initial transactions only if status indicates data might exist
        if (status.has_flagged_data || status.simulation_running) {
            await fetchInitialTransactions();
        } else {
            setLoading((prev) => ({ ...prev, transactions: false }));
        }
        isInitialLoadDone.current = true; // Mark initial load as done
         console.log("Initial data load complete.");
    };

    performInitialLoad();

    // --- WebSocket Event Handlers ---
    const onConnect = () => {
      console.log('Socket connected');
      setStatus(prev => ({ ...prev, websocket_connected: true }));
      showNotification('Real-time connection established.', 'success', 3000);
      fetchStatus(); // Refresh status from backend upon connection
    };

    const onDisconnect = (reason) => {
      console.log('Socket disconnected:', reason);
      setStatus(prev => ({ ...prev, websocket_connected: false, simulation_running: false })); // Assume simulation stops if socket drops
      showNotification('Real-time connection lost.', 'error');
    };

    const onNewTransaction = (newTransaction) => {
      console.log('Received new flagged transaction:', newTransaction);
      setTransactions(prev => {
          // Add to start, maintain sort order (newest first), and limit size
          const updated = [newTransaction, ...prev];
          return updated.slice(0, MAX_TRANSACTIONS);
      });
      // Maybe a less intrusive notification?
      // showNotification(`New tx flagged: ${newTransaction.tx_hash.substring(0, 10)}...`, 'warning', 3000);
    };

    const onTransactionUpdate = (updatedTransaction) => {
       console.log('Received transaction update:', updatedTransaction);
       setTransactions(prev => prev.map(tx =>
          tx.tx_hash === updatedTransaction.tx_hash ? updatedTransaction : tx
       ));
       showNotification(`Tx ${updatedTransaction.tx_hash.substring(0,10)}... updated.`, 'info', 3000);
    };

    const onThreatListUpdate = (updatedList) => {
        console.log('Received threat list update');
        setThreatList(updatedList);
        showNotification('Threat list updated.', 'info', 3000);
    };

    // --- Connect and Register Listeners ---
    if (!socket.connected) {
      console.log("Attempting to connect socket...");
      socket.connect();
    }
    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);
    socket.on('new_flagged_transaction', onNewTransaction);
    socket.on('transaction_update', onTransactionUpdate);
    socket.on('threat_list_updated', onThreatListUpdate); // Listener for threat list updates

    // --- Cleanup Function ---
    return () => {
      console.log("Cleaning up socket listeners and disconnecting...");
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('new_flagged_transaction', onNewTransaction);
      socket.off('transaction_update', onTransactionUpdate);
      socket.off('threat_list_updated', onThreatListUpdate);
      // Only disconnect if you want it to stop when the component unmounts
      // If App is the main component, maybe keep it connected?
      // socket.disconnect();
      // setStatus(prev => ({ ...prev, websocket_connected: false }));
    };
    // Re-run effect minimally, only on mount/unmount in this setup
    // Dependencies like showNotification, fetchStatus should be stable via useCallback
  }, [showNotification, fetchStatus, fetchThreatList, fetchInitialTransactions]); // Add fetch callbacks to deps


  return {
    status,
    transactions,
    loading,
    error,
    notification,
    runSetup, // Keep setup action
    // runSimulation, // Removed
    fetchTransactions: fetchInitialTransactions, // Rename for clarity (fetches initial only)
    clearNotification,
    threatList,
    selectedWalletProfile,
    fetchThreatList, // Keep for potential manual refresh
    addWalletToThreats,
    removeWalletFromThreats,
    submitTxReview,
    fetchWalletDetails,
  };
};

export default useDashboard;