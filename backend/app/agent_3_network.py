"""
Agent 3: Enhanced Network Analysis Engine
- Real-time graph analysis with temporal awareness
- Detects circular transfers, smurfing, collection patterns
- Velocity-based anomaly detection
"""
from typing import Dict, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta
import networkx as nx

# --- Global State for Agent 3 ---
G = nx.DiGraph()  # Main transaction graph

# Transaction history (stores: (from, to, value, timestamp))
transaction_log: Deque[tuple] = deque(maxlen=10000)  # Reduced for better performance

# Velocity tracking: wallet -> list of (timestamp, value) tuples
wallet_velocity = defaultdict(lambda: deque(maxlen=50))

# Address interaction tracking
address_interactions = defaultdict(lambda: {
    'out_edges': set(),  # Addresses this wallet sent to
    'in_edges': set(),   # Addresses that sent to this wallet
    'total_out': 0.0,    # Total ETH sent
    'total_in': 0.0,     # Total ETH received
    'last_activity': None
})

# Time window for velocity checks (in seconds)
VELOCITY_WINDOW = 600  # 10 minutes
RAPID_TX_THRESHOLD = 8  # Number of txs in window to flag
MICRO_VALUE_THRESHOLD = 0.1  # ETH


def agent_3_network(transaction: Dict) -> Dict:
    """
    Agent 3: Real-time Network Analysis with Enhanced Pattern Detection
    """
    global G, transaction_log, wallet_velocity, address_interactions

    score = 0
    reasons = []

    try:
        from_addr = transaction.get('from_address', '').lower()
        to_addr = transaction.get('to_address', '').lower()
        value = transaction.get('value_eth', 0.0)
        
        # Parse timestamp
        try:
            timestamp_str = transaction.get('timestamp', '')
            current_time = datetime.fromisoformat(timestamp_str.replace('Z', ''))
        except:
            current_time = datetime.now()

        # Skip self-transfers or invalid addresses
        if not from_addr or not to_addr or from_addr == to_addr:
            transaction['agent_3_score'] = 0
            transaction['agent_3_reasons'] = ["Network analysis skipped (invalid addresses)"]
            return transaction

        # --- 1. PRUNE OLD TRANSACTIONS ---
        if len(transaction_log) == transaction_log.maxlen:
            old_from, old_to, old_value, old_time = transaction_log[0]
            
            # Remove edge from graph
            if G.has_edge(old_from, old_to):
                G.remove_edge(old_from, old_to)
                
                # Clean up isolated nodes
                if G.degree(old_from) == 0:
                    G.remove_node(old_from)
                    if old_from in address_interactions:
                        del address_interactions[old_from]
                        
                if G.degree(old_to) == 0:
                    G.remove_node(old_to)
                    if old_to in address_interactions:
                        del address_interactions[old_to]

        # --- 2. ADD NEW TRANSACTION TO GRAPH ---
        G.add_edge(from_addr, to_addr, weight=value, timestamp=current_time)
        transaction_log.append((from_addr, to_addr, value, current_time))
        
        # Update interaction tracking
        address_interactions[from_addr]['out_edges'].add(to_addr)
        address_interactions[from_addr]['total_out'] += value
        address_interactions[from_addr]['last_activity'] = current_time
        
        address_interactions[to_addr]['in_edges'].add(from_addr)
        address_interactions[to_addr]['total_in'] += value
        address_interactions[to_addr]['last_activity'] = current_time

        # --- 3. VELOCITY TRACKING ---
        wallet_velocity[from_addr].append((current_time, value))
        
        # Clean old velocity entries
        cutoff_time = current_time - timedelta(seconds=VELOCITY_WINDOW)
        while wallet_velocity[from_addr] and wallet_velocity[from_addr][0][0] < cutoff_time:
            wallet_velocity[from_addr].popleft()

        # --- 4. PATTERN DETECTION ---

        # Check 1: Circular Transfer Detection (A -> ... -> A)
        # More sophisticated: Check if a return path exists
        if nx.has_path(G, to_addr, from_addr):
            try:
                path_length = nx.shortest_path_length(G, source=to_addr, target=from_addr)
                
                if path_length == 1:
                    # Direct back-and-forth (A -> B -> A)
                    score += 45
                    reasons.append(f"Direct circular transfer detected (2-hop cycle)")
                elif path_length <= 3:
                    # Short cycle (3-4 hops)
                    score += 35
                    reasons.append(f"Circular transfer pattern detected ({path_length + 1}-hop cycle)")
                elif path_length <= 5:
                    # Medium cycle (4-6 hops)
                    score += 25
                    reasons.append(f"Potential circular path detected ({path_length + 1} hops)")
            except nx.NetworkXNoPath:
                pass

        # Check 2: Fan-Out / Smurfing Detection
        # Sender distributing funds to many addresses
        sender_out_degree = G.out_degree(from_addr)
        sender_unique_recipients = len(address_interactions[from_addr]['out_edges'])
        
        if sender_unique_recipients >= 20:
            score += 35
            reasons.append(f"High fan-out: {sender_unique_recipients} unique recipients (smurfing pattern)")
        elif sender_unique_recipients >= 12:
            score += 20
            reasons.append(f"Moderate fan-out: {sender_unique_recipients} unique recipients")

        # Check 3: Fan-In / Collection Detection
        # Receiver collecting funds from many addresses
        receiver_in_degree = G.in_degree(to_addr)
        receiver_unique_senders = len(address_interactions[to_addr]['in_edges'])
        
        if receiver_unique_senders >= 20:
            score += 35
            reasons.append(f"High fan-in: {receiver_unique_senders} unique senders (collection pattern)")
        elif receiver_unique_senders >= 12:
            score += 20
            reasons.append(f"Moderate fan-in: {receiver_unique_senders} unique senders")

        # Check 4: Rapid Transaction Velocity
        recent_txs = len(wallet_velocity[from_addr])
        if recent_txs >= RAPID_TX_THRESHOLD:
            total_recent_value = sum(v for _, v in wallet_velocity[from_addr])
            avg_value = total_recent_value / recent_txs
            
            if avg_value < MICRO_VALUE_THRESHOLD:
                score += 30
                reasons.append(f"Rapid micro-transactions: {recent_txs} txs in {VELOCITY_WINDOW//60} minutes (avg: {avg_value:.4f} ETH)")
            else:
                score += 15
                reasons.append(f"High transaction velocity: {recent_txs} txs in {VELOCITY_WINDOW//60} minutes")

        # Check 5: Balanced In/Out (Money Laundering Pattern)
        # If sender has received similar amount to what they're sending out
        if from_addr in address_interactions:
            total_in = address_interactions[from_addr]['total_in']
            total_out = address_interactions[from_addr]['total_out']
            
            if total_in > 0 and total_out > 0:
                ratio = min(total_in, total_out) / max(total_in, total_out)
                
                # If in/out are very balanced (typical of pass-through accounts)
                if ratio > 0.85 and sender_unique_recipients >= 5:
                    score += 20
                    reasons.append(f"Pass-through pattern: Balanced in/out ratio ({ratio:.2f}) with multiple recipients")

        # Check 6: New Address Sending Large Amount
        # If this is one of the first transactions from an address
        if sender_out_degree <= 3 and value > 10.0:
            score += 15
            reasons.append(f"New address sending large amount: {value:.2f} ETH (only {sender_out_degree} prior txs)")

        # Check 7: Hub Detection (Mixing Service Pattern)
        # Very high degree in both directions
        if sender_out_degree > 30 and receiver_in_degree > 30:
            score += 25
            reasons.append(f"Potential mixer/hub detected: High bidirectional traffic")

        # --- 5. FINALIZE ---
        if not reasons:
            reasons = ["No network anomalies detected"]
        
        # Cap the score at 100 for this agent
        score = min(score, 100)
        
        print(f"  [Agent 3] Network analysis complete. Score: {score}")

    except Exception as e:
        print(f"  [Agent 3] ERROR in network analysis: {e}")
        import traceback
        traceback.print_exc()
        score = 0
        reasons = [f"Network analysis failed: {str(e)}"]

    # Add score and reasons to transaction
    transaction['agent_3_score'] = score
    transaction['agent_3_reasons'] = reasons

    return transaction


def get_network_stats():
    """
    Debug function to get current network statistics
    """
    return {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'transactions_tracked': len(transaction_log),
        'addresses_tracked': len(address_interactions)
    }


def reset_network_state():
    """
    Reset all network state (useful for testing)
    """
    global G, transaction_log, wallet_velocity, address_interactions
    G.clear()
    transaction_log.clear()
    wallet_velocity.clear()
    address_interactions.clear()
    print("[Agent 3] Network state reset")