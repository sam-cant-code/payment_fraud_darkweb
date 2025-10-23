"""
Agent 3: Network Analysis Engine (Live Implementation)
- Builds a stateful, in-memory graph of recent transactions.
- Analyzes network patterns in real-time.
- Detects circular transfers and fan-in/fan-out (smurfing/collection).
"""
from typing import Dict, Deque
from collections import deque
from datetime import datetime, timedelta
import networkx as nx

# --- Global State for Agent 3 ---
# A directed graph to store recent transaction flows
G = nx.DiGraph()

# A deque to remember the order of transactions for pruning
# We'll keep the last 20,000 edges (approx 10-15k nodes)
transaction_log: Deque[tuple] = deque(maxlen=20000)
# --- End Global State ---


def agent_3_network(transaction: Dict) -> Dict:
    """
    Agent 3: Real-time Network Analysis
    - Adds the current transaction to the graph.
    - Prunes the oldest transaction from the graph.
    - Checks for circular transfers and fan-in/fan-out patterns.
    """
    global G, transaction_log

    score = 0
    reasons = []

    try:
        from_addr = transaction.get('from_address', '').lower()
        to_addr = transaction.get('to_address', '').lower()

        if not from_addr or not to_addr or from_addr == to_addr:
            transaction['agent_3_score'] = 0
            transaction['agent_3_reasons'] = ["Network analysis skipped (self-transfer or missing address)"]
            return transaction

        # --- 1. Prune Graph ---
        # If the log is full, remove the oldest edge from the graph to keep it fresh
        if len(transaction_log) == transaction_log.maxlen:
            old_from, old_to = transaction_log[0] # Get the oldest edge
            if G.has_edge(old_from, old_to):
                G.remove_edge(old_from, old_to)
                # Optional: Prune nodes if they have no more edges
                if G.degree(old_from) == 0:
                    G.remove_node(old_from)
                if G.degree(old_to) == 0:
                    G.remove_node(old_to)

        # --- 2. Add New Transaction to Graph and Log ---
        G.add_edge(from_addr, to_addr)
        transaction_log.append((from_addr, to_addr))

        # --- 3. Run Network-Based Checks ---

        # Check 1: Circular Transfer (A -> B ... -> A)
        # Check if a path exists from the *recipient* back to the *sender*
        if nx.has_path(G, to_addr, from_addr):
            try:
                # Find the shortest path back
                path_len = nx.shortest_path_length(G, source=to_addr, target=from_addr)
                if path_len < 6: # e.g., A->B->C->A (length 2 path back)
                    score += 40
                    reasons.append(f"Suspicious circular transfer risk: Path of length {path_len+1} detected (A...->A)")
            except nx.NetworkXNoPath:
                pass # Should be caught by has_path, but good to be safe

        # Check 2: Fan-Out / Smurfing (A -> B, A -> C, A -> D...)
        # Sender has a high number of unique, recent recipients
        sender_out_degree = G.out_degree(from_addr)
        if sender_out_degree > 15:
            score += 25
            reasons.append(f"High fan-out (smurfing risk): Sender has {sender_out_degree} unique recipients in recent history.")

        # Check 3: Fan-In / Collection (B -> A, C -> A, D -> A...)
        # Receiver has a high number of unique, recent senders
        receiver_in_degree = G.in_degree(to_addr)
        if receiver_in_degree > 15:
            score += 25
            reasons.append(f"High fan-in (collection risk): Receiver has {receiver_in_degree} unique senders in recent history.")

        if not reasons:
            reasons = ["No immediate network threats detected"]
        
        print(f"  [Agent 3] Network analysis complete. Score: {score}")


    except Exception as e:
        print(f"  [Agent 3] ERROR in network analysis: {e}")
        score = 0
        reasons = [f"Network analysis failed: {e}"]

    # Add score and reasons
    transaction['agent_3_score'] = score
    transaction['agent_3_reasons'] = reasons

    return transaction