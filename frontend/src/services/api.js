import axios from 'axios';
import { io } from 'socket.io-client';

// Define the BASE URL (e.g., http://127.0.0.1:5000)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000';

const apiClient = axios.create({
  // Set baseURL to ONLY the host and port
  baseURL: API_BASE_URL, // Corrected: Should be e.g., http://127.0.0.1:5000
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- API Functions ---
// Prepend '' to every request path
export const getStatus = () => apiClient.get('/status');
export const postRunSetup = () => apiClient.post('/setup');
export const getFlaggedTransactions = () => apiClient.get('/flagged-transactions');
export const getWalletProfile = (address) => apiClient.get(`/wallet/${address}`);
export const postReview = (txHash, status, analystId = null) => {
  const payload = { status };
  if (analystId) payload.analyst_id = analystId;
  return apiClient.post(`/review/${txHash}`, payload);
};
export const getThreats = () => apiClient.get('/threats');
export const postThreat = (walletAddress) => apiClient.post('/threats', { wallet_address: walletAddress });
export const deleteThreat = (walletAddress) => apiClient.delete('/threats', { data: { wallet_address: walletAddress } });


// --- WebSocket Connection ---
// Socket connects to the BASE URL (without ) - this remains correct
export const socket = io(API_BASE_URL, {
    autoConnect: false,
    transports: ['websocket']
});

// Log socket events for debugging (optional)
socket.onAny((event, ...args) => {
  console.log("Socket Event:", event, args);
});