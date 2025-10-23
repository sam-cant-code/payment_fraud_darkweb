import { useState, useEffect, useCallback, useRef } from 'react'; // Added useRef
import {
  getStatus,
  postRunSetup,
  // postRunSimulation, // Removed
  getFlaggedTransactions, // Still needed for initial load (now gets all tx)
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
    simulation_running: false, // Changed from simulation_run -> listener_active
    threat_list_available: false,
    has_flagged_data: false, // Added from backend status (now means CSV has data)
    websocket_connected: false, // Added connection status
    listener_active: false, // Tracks the Alchemy listener status from backend
  });
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState({
    status: true,
    setup: false,
    // simulation: false, // Removed
    transactions: false, // For initial load - SET TO FALSE
    review: false,
    threats: true, // Load threats initially
    walletProfile: false,
  });
  const [error, setError] = useState({ status: null, setup: null, transactions: null, review: null, threats: null, walletProfile: null }); // Expanded error state
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
      // Ensure backend response structure matches expected state keys
      const backendStatus = response.data;
      setStatus(prevStatus => ({
          ...prevStatus, // Keep existing websocket_connected status
          database_ready: backendStatus.database_ready ?? false,
          simulation_running: backendStatus.simulation_running ?? false, // Keep for Sidebar logic if needed
          listener_active: backendStatus.listener_active ?? false, // Use specific listener status
          threat_list_available: backendStatus.threat_list_available ?? false,
          has_flagged_data: backendStatus.has_flagged_data ?? false,
          // Do not update websocket_connected from here, handled by socket events
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
  }, [showNotification]); // Added missing dependency

  // Fetches the *initial* list of transactions from the CSV/DB log (now includes all statuses)
  // THIS FUNCTION IS NO LONGER CALLED ON LOAD
  const fetchInitialTransactions = useCallback(async () => {
    setLoading((prev) => ({ ...prev, transactions: true }));
    setError((prev) => ({ ...prev, transactions: null }));
    try {
      const response = await getFlaggedTransactions(); // API endpoint name is kept, but it returns all now
      // Sort by timestamp descending for initial view (newest first)
      const sortedTransactions = response.data.sort((a, b) =>
        new Date(b.timestamp) - new Date(a.timestamp)
      );
      setTransactions(sortedTransactions.slice(0, MAX_TRANSACTIONS)); // Apply limit
      console.log(`Loaded initial ${sortedTransactions.length} transactions (all statuses).`);
    } catch (err) {
       if (err.response && err.response.status === 404) {
        setTransactions([]);
        console.log("Initial transaction file not found or empty (API returned 404).");
        // Don't show error notification for 404, it's a valid state if file is empty/missing
      } else {
        setError((prev) => ({ ...prev, transactions: 'Failed to fetch initial transactions' }));
        showNotification('Failed to fetch initial transactions.', 'error');
        console.error("Initial transaction fetch error:", err);
      }
    } finally {
      setLoading((prev) => ({ ...prev, transactions: false }));
    }
  }, [showNotification]); // Added missing dependency

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
  }, [showNotification]); // Added missing dependency

  // --- Action Callbacks ---
  const runSetup = useCallback(async () => {
    setLoading((prev) => ({ ...prev, setup: true }));
    setError((prev) => ({ ...prev, setup: null }));
    try {
      const response = await postRunSetup();
      showNotification(response.data.message, 'success');
      // After setup, fetch everything again to reflect new state
      await fetchStatus();
      await fetchThreatList();
      // We still don't fetch initial transactions here, wait for live ones
      // await fetchInitialTransactions(); // <-- REMAINS COMMENTED
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Setup failed';
      setError((prev) => ({ ...prev, setup: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Setup error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, setup: false }));
    }
    // Added fetchThreatList, removed fetchInitialTransactions from dependencies
  }, [fetchStatus, showNotification, fetchThreatList]);

  // runSimulation is no longer needed as a direct user action

  const submitTxReview = useCallback(async (txHash, newStatus) => {
    setLoading((prev) => ({ ...prev, review: true }));
    setError((prev) => ({ ...prev, review: null }));
    try {
      const response = await postReview(txHash, newStatus);
      showNotification(response.data.message, 'success');
      // No immediate local state update needed; rely on 'transaction_update' event
    } catch (err) {
      const errorMsg = err.response?.data?.error || 'Review submission failed';
      setError((prev) => ({ ...prev, review: errorMsg }));
      showNotification(errorMsg, 'error');
      console.error("Review error:", err);
    } finally {
      setLoading((prev) => ({ ...prev, review: false }));
    }
  }, [showNotification]); // Added missing dependency

  // addWalletToThreats & removeWalletFromThreats rely on 'threat_list_updated' event
 const addWalletToThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true })); // Indicate loading
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await postThreat(walletAddress);
        showNotification(`Attempting to add ${walletAddress.substring(0,10)}...`, 'info', 2000); // Give feedback
        // Backend now emits 'threat_list_updated', no local update needed here
    } catch (err) {
       const errorMsg = err.response?.data?.error || 'Failed to add wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Add threat error:", err);
       setLoading((prev) => ({ ...prev, threats: false })); // Stop loading on error
    }
    // Loading state reset by 'threat_list_updated' event handler or error handler
}, [showNotification]); // Removed fetchThreatList

const removeWalletFromThreats = useCallback(async (walletAddress) => {
    setLoading((prev) => ({ ...prev, threats: true })); // Indicate loading
    setError((prev) => ({ ...prev, threats: null }));
    try {
        await deleteThreat(walletAddress);
        showNotification(`Attempting to remove ${walletAddress.substring(0,10)}...`, 'info', 2000); // Give feedback
         // Backend now emits 'threat_list_updated', no local update needed here
    } catch (err) {
       const errorMsg = err.response?.data?.error || 'Failed to remove wallet';
       setError((prev) => ({ ...prev, threats: errorMsg }));
       showNotification(errorMsg, 'error');
       console.error("Remove threat error:", err);
       setLoading((prev) => ({ ...prev, threats: false })); // Stop loading on error
    }
     // Loading state reset by 'threat_list_updated' event handler or error handler
}, [showNotification]); // Removed fetchThreatList

  // Fetch Wallet Details (example, might need implementation)
  const fetchWalletDetails = useCallback(async (address) => {
    if (!address) return;
    setLoading(prev => ({ ...prev, walletProfile: true }));
    setError(prev => ({ ...prev, walletProfile: null }));
    setSelectedWalletProfile(null); // Clear previous
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
        console.log("Performing initial data load (SKIPPING transactions)...");
        // Only set loading for status and threats
        setLoading(prev => ({ ...prev, status: true, threats: true, transactions: false }));
        await fetchStatus();
        await fetchThreatList();
        
        // --- MODIFICATION ---
        // We no longer fetch initial transactions to show only live data
        // await fetchInitialTransactions(); 
        console.log("Skipped initial transaction load.");
        // --- END MODIFICATION ---

        isInitialLoadDone.current = true; // Mark initial load as done
        // Clear initial loads
        setLoading(prev => ({ ...prev, status: false, threats: false, transactions: false }));
        console.log("Initial data load complete (status and threats only).");
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
      setStatus(prev => ({ ...prev, websocket_connected: false, listener_active: false })); // Assume listener stops if socket drops
      showNotification('Real-time connection lost.', 'error');
    };

    // This function handles incoming transactions from the websocket
    const onNewTransaction = (newTransaction) => {
      console.log('Received new scanned transaction:', newTransaction); // Log changed slightly
      setTransactions(prev => {
          // Add to start, maintain sort order (newest first), and limit size
          // Ensure no duplicates if initial load races with websocket
          if (prev.some(tx => tx.tx_hash === newTransaction.tx_hash)) {
              console.log(`Tx ${newTransaction.tx_hash.substring(0,10)}... already exists.`);
              return prev; // Already exists, do nothing
          }
          const updated = [newTransaction, ...prev];
          return updated.slice(0, MAX_TRANSACTIONS);
      });
      // Optional: Show a subtle notification for approved ones?
       if (newTransaction.final_status === 'APPROVE') {
         // Maybe too noisy - consider removing or making it very subtle
         // showNotification(`Tx ${newTransaction.tx_hash.substring(0,10)}... approved.`, 'info', 2000);
       } else {
           // Highlight Flagged/Denied ones
           const messageType = newTransaction.final_status === 'DENY' ? 'error' : 'warning';
           showNotification(`Tx ${newTransaction.tx_hash.substring(0,10)}... ${newTransaction.final_status.replace('_',' ')}.`, messageType, 4000);
       }
    };

    const onTransactionUpdate = (updatedTransaction) => {
       console.log('Received transaction update:', updatedTransaction);
       setTransactions(prev => prev.map(tx =>
          tx.tx_hash === updatedTransaction.tx_hash ? updatedTransaction : tx
       ));
       showNotification(`Tx ${updatedTransaction.tx_hash.substring(0,10)}... review updated.`, 'info', 3000);
    };

    const onThreatListUpdate = (updatedList) => {
        console.log('Received threat list update');
        setThreatList(updatedList);
        setLoading((prev) => ({ ...prev, threats: false })); // Stop loading indicator after update
        showNotification('Threat list updated.', 'info', 3000);
    };

    // --- Connect and Register Listeners ---
    if (!socket.connected) {
      console.log("Attempting to connect socket...");
      socket.connect();
    }
    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);

    // Listen for ALL scanned transactions, not just flagged ones
    socket.on('new_scanned_transaction', onNewTransaction);

    socket.on('transaction_update', onTransactionUpdate);
    socket.on('threat_list_updated', onThreatListUpdate); // Listener for threat list updates

    // --- Cleanup Function ---
    return () => {
      console.log("Cleaning up socket listeners...");
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('new_scanned_transaction', onNewTransaction); // Clean up the correct listener
      socket.off('transaction_update', onTransactionUpdate);
      socket.off('threat_list_updated', onThreatListUpdate);
      // Decide whether to disconnect on unmount based on app structure
      // If this hook is in the main App component, maybe leave it connected.
      // socket.disconnect();
      // setStatus(prev => ({ ...prev, websocket_connected: false }));
    };
    // Ensure all useCallback-wrapped functions used in the effect are listed
  }, [showNotification, fetchStatus, fetchThreatList]); // Removed fetchInitialTransactions from dependency array


  return {
    status,
    transactions,
    loading,
    error,
    notification,
    runSetup, // Keep setup action
    // runSimulation, // Removed
    fetchTransactions: fetchInitialTransactions, // Export renamed function (can still be called manually if needed)
    clearNotification,
    threatList,
    selectedWalletProfile, // Export selected profile state
    fetchThreatList, // Keep for potential manual refresh
    addWalletToThreats,
    removeWalletFromThreats,
    submitTxReview,
    fetchWalletDetails, // Export wallet detail fetch function
  };
};

export default useDashboard;