import axios from 'axios';

// Use the environment variable, with a fallback
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:5000/api';

const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Fetches the current backend status.
 */
export const getStatus = () => {
  return apiClient.get('/status');
};

/**
 * Runs the one-time database setup script.
 */
export const postRunSetup = () => {
  return apiClient.post('/setup');
};

/**
 * Runs the simulation script.
 */
export const postRunSimulation = () => {
  return apiClient.post('/run-simulation');
};

/**
 * Fetches the list of flagged transactions.
 */
export const getFlaggedTransactions = () => {
  return apiClient.get('/flagged-transactions');
};