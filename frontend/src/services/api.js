// frontend/src/services/api.js
import axios from 'axios';
import { io } from 'socket.io-client';

// Define the BASE URL (e.g., http://127.0.0.1:5000)
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api'; // Corrected to include /api

// Socket.IO connects to base URL (without /api)
// Extract the base URL without any path suffix for socket connection
const SOCKET_BASE_URL = API_BASE_URL.replace(/\/api\/?$/, '');

console.log('API Base URL:', API_BASE_URL);
console.log('Socket Base URL:', SOCKET_BASE_URL);

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// --- API Functions ---
export const getStatus = () => apiClient.get('/status');
export const postRunSetup = () => apiClient.post('/setup');
export const getFlaggedTransactions = () => apiClient.get('/flagged-transactions');
export const getWalletProfile = (address) => apiClient.get(`/wallet-profile/${address}`); // Updated path
export const postReview = (txHash, status, analystId = null) => {
  const payload = { status };
  if (analystId) payload.analyst_id = analystId;
  return apiClient.post(`/review/${txHash}`, payload);
};
export const getThreats = () => apiClient.get('/threats');
export const postThreat = (walletAddress) => apiClient.post('/threats', { wallet_address: walletAddress });
export const deleteThreat = (walletAddress) => apiClient.delete('/threats', { data: { wallet_address: walletAddress } });

// --- NEW SIMULATION FUNCTION ---
export const postSimulateTransaction = (txData) => {
  // txData should be { from_address, to_address, value_eth, gas_price }
  return apiClient.post('/simulate-transaction', txData);
};


// --- WebSocket Connection (No changes) ---
export const socket = io(SOCKET_BASE_URL, {
    autoConnect: false,
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 1000,
    reconnectionDelayMax: 5000,
    reconnectionAttempts: 5,
    timeout: 10000,
});

socket.on('connect_error', (error) => {
  console.error('Socket connection error:', error.message);
  console.error('Is the backend running on', SOCKET_BASE_URL, '?');
});

socket.on('connect_timeout', () => {
  console.error('Socket connection timeout');
});

socket.onAny((event, ...args) => {
  console.log("Socket Event:", event, args);
});