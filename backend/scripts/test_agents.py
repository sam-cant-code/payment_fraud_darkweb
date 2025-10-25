"""
Agent Testing Suite for Blockchain Fraud Detection MVP
Tests ALL agents individually with calibrated thresholds
FIXED: Uses actual wallet addresses from the threat list
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, UTC

# Fix the path to import from app directory
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

# Add both backend and app to path
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(BACKEND_DIR / 'app'))

# Now import agents
from app.agents import (
    agent_1_monitor,
    agent_2_behavioral, 
    agent_4_decision,
    get_system_thresholds
)

# Import agent 3 directly
try:
    from app.agent_3_network import agent_3_network
    AGENT_3_AVAILABLE = True
    print("[test_agents.py] ✓ Agent 3 (Network Analysis) loaded successfully")
except ImportError as e:
    print(f"[test_agents.py] ⚠️  Warning: Could not import agent_3_network: {e}")
    AGENT_3_AVAILABLE = False
    def agent_3_network(transaction):
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction


# --- LOAD REAL THREAT WALLETS ---
def load_real_threat_wallets():
    """Load actual wallets from the threat list"""
    threat_file = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'
    if not threat_file.exists():
        print("⚠️  WARNING: Threat list not found!")
        return [], []
    
    with open(threat_file, 'r') as f:
        all_threats = [line.strip().lower() for line in f if line.strip()]
    
    # Separate malicious and mixers
    malicious = [w for w in all_threats if w.startswith('0xbad')]
    mixers = [w for w in all_threats if w.startswith('0xmix')]
    
    print(f"[test_agents.py] Loaded {len(malicious)} malicious + {len(mixers)} mixer wallets from threat list")
    return malicious, mixers

MALICIOUS_WALLETS, MIXER_WALLETS = load_real_threat_wallets()


def reset_agent_3_state():
    """Reset Agent 3's global graph state"""
    if AGENT_3_AVAILABLE:
        try:
            from app.agent_3_network import G, transaction_log
            G.clear()
            transaction_log.clear()
            print("[Agent 3] Network state reset")
        except Exception as e:
            print(f"Warning: Could not reset Agent 3 state: {e}")


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"           {title}")
    print("="*70)


def print_result(transaction, test_name=""):
    """Print test result in a clean format"""
    status = transaction.get('final_status', 'N/A')
    score = transaction.get('final_score', 0)
    
    print(f"\n{test_name}")
    print(f"  Status: {status}")
    print(f"  Final Score: {score}/150")
    print(f"  Agent Breakdown:")
    print(f"    Agent 1 (Threat Intel): {transaction.get('agent_1_score', 0)}")
    print(f"    Agent 2 (ML Behavioral): {transaction.get('agent_2_score', 0)}")
    print(f"    Agent 3 (Network): {transaction.get('agent_3_score', 0)}")
    
    # Show ML probability if available
    ml_prob = transaction.get('ml_fraud_probability', -1)
    if ml_prob >= 0:
        print(f"    ML Fraud Probability: {ml_prob:.1%}")
    
    # Combine all reasons
    reasons = []
    for key in ['agent_1_reasons', 'agent_2_reasons', 'agent_3_reasons']:
        agent_reasons = transaction.get(key, [])
        if isinstance(agent_reasons, list):
            reasons.extend(agent_reasons)
    
    if reasons:
        print(f"  Reasons:")
        for reason in reasons:
            if reason and not reason.lower().startswith("no"):
                print(f"    - {reason}")
    
    return transaction


def test_agent_1_threat_intel():
    """Test Agent 1: Threat Intelligence"""
    print_header("TEST 1: THREAT INTELLIGENCE (Agent 1)")
    
    if not MALICIOUS_WALLETS or not MIXER_WALLETS:
        print("⚠️  WARNING: No threat wallets loaded. Skipping Agent 1 tests.")
        return
    
    print(f"Testing with {len(MALICIOUS_WALLETS)} malicious + {len(MIXER_WALLETS)} mixer wallets\n")
    
    base_time = datetime.now(UTC)
    
    # Test 1.1: Normal transaction
    tx_normal = {
        "tx_hash": "0xtest_normal_001",
        "from_address": "0xnorm000000000000000000000000000000000000",
        "to_address": "0xnorm000000000000000000000000000000000001",
        "value_eth": 5.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_normal = agent_1_monitor(tx_normal)
    tx_normal = agent_2_behavioral(tx_normal)
    tx_normal = agent_3_network(tx_normal)
    tx_normal = agent_4_decision(tx_normal)
    print_result(tx_normal, "1.1 Normal Transaction:")
    
    # Test 1.2: Malicious sender (USE REAL WALLET)
    tx_malicious_sender = {
        "tx_hash": "0xtest_malicious_002",
        "from_address": MALICIOUS_WALLETS[0],  # ✅ USE REAL WALLET
        "to_address": "0xnorm000000000000000000000000000000000002",
        "value_eth": 10.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_malicious_sender = agent_1_monitor(tx_malicious_sender)
    tx_malicious_sender = agent_2_behavioral(tx_malicious_sender)
    tx_malicious_sender = agent_3_network(tx_malicious_sender)
    tx_malicious_sender = agent_4_decision(tx_malicious_sender)
    print_result(tx_malicious_sender, "1.2 Malicious Sender:")
    
    if tx_malicious_sender.get('agent_1_score', 0) > 0:
        print("  ✅ Agent 1 detected malicious sender")
    else:
        print("  ❌ Agent 1 FAILED to detect malicious sender")
    
    # Test 1.3: Mixer transaction (USE REAL MIXER)
    tx_mixer = {
        "tx_hash": "0xtest_mixer_003",
        "from_address": "0xnorm000000000000000000000000000000000003",
        "to_address": MIXER_WALLETS[0],  # ✅ USE REAL MIXER
        "value_eth": 8.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_mixer = agent_1_monitor(tx_mixer)
    tx_mixer = agent_2_behavioral(tx_mixer)
    tx_mixer = agent_3_network(tx_mixer)
    tx_mixer = agent_4_decision(tx_mixer)
    print_result(tx_mixer, "1.3 Mixer Transaction:")
    
    if tx_mixer.get('agent_1_score', 0) >= 50:
        print("  ✅ Agent 1 detected mixer (score >= 50)")
    else:
        print(f"  ❌ Agent 1 mixer score too low: {tx_mixer.get('agent_1_score', 0)}")
    
    # Test 1.4: High value
    tx_high_value = {
        "tx_hash": "0xtest_highval_004",
        "from_address": "0xwhale00000000000000000000000000000000000",
        "to_address": "0xnorm000000000000000000000000000000000004",
        "value_eth": 150.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_high_value = agent_1_monitor(tx_high_value)
    tx_high_value = agent_2_behavioral(tx_high_value)
    tx_high_value = agent_3_network(tx_high_value)
    tx_high_value = agent_4_decision(tx_high_value)
    print_result(tx_high_value, "1.4 High Value Transaction:")
    
    # Test 1.5: Structuring (rapid small transfers)
    print("\n1.5 Rapid Small Transfers (Structuring):")
    sender = "0xnorm000000000000000000000000000000000005"
    current_time = base_time
    
    for i in range(6):
        current_time += timedelta(minutes=1)
        tx_struct = {
            "tx_hash": f"0xtest_struct_{i:03d}",
            "from_address": sender,
            "to_address": f"0xnorm00000000000000000000000000000000000{i+6}",
            "value_eth": 0.03,
            "gas_price": 50.0,
            "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
        }
        tx_struct = agent_1_monitor(tx_struct)
        tx_struct = agent_2_behavioral(tx_struct)
        tx_struct = agent_3_network(tx_struct)
        tx_struct = agent_4_decision(tx_struct)
        
        if i == 5:  # Print last one
            print_result(tx_struct, f"\n  Transfer #{i+1}:")
            if tx_struct.get('agent_1_score', 0) >= 15:
                print("  ✅ Agent 1 detected structuring pattern")
    
    print("\n✓ Agent 1 Tests Complete")


def test_agent_2_ml_behavioral():
    """Test Agent 2: ML Behavioral Analysis"""
    print_header("TEST 2: ML BEHAVIORAL ANALYSIS (Agent 2)")
    
    if not MALICIOUS_WALLETS or not MIXER_WALLETS:
        print("⚠️  WARNING: No threat wallets loaded. Skipping some tests.")
        return
    
    base_time = datetime.now(UTC)
    
    # Test 2.1: Transaction that should trigger ML model (known fraud pattern)
    print("\n2.1 Known Fraud Pattern (Malicious -> High Value):")
    tx_fraud_pattern = {
        "tx_hash": "0xtest_ml_fraud_001",
        "from_address": MALICIOUS_WALLETS[1] if len(MALICIOUS_WALLETS) > 1 else MALICIOUS_WALLETS[0],
        "to_address": "0xnorm000000000000000000000000000000000010",
        "value_eth": 85.0,
        "gas_price": 200.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_fraud_pattern = agent_1_monitor(tx_fraud_pattern)
    tx_fraud_pattern = agent_2_behavioral(tx_fraud_pattern)
    tx_fraud_pattern = agent_3_network(tx_fraud_pattern)
    tx_fraud_pattern = agent_4_decision(tx_fraud_pattern)
    print_result(tx_fraud_pattern)
    
    if tx_fraud_pattern.get('agent_2_score', 0) > 0:
        print("  ✅ Agent 2 detected risk via ML model")
    else:
        print("  ⚠️  Agent 2 did not flag this transaction")
    
    # Test 2.2: Normal transaction
    print("\n2.2 Normal Transaction (Low Risk):")
    tx_normal = {
        "tx_hash": "0xtest_ml_normal_001",
        "from_address": "0xnorm000000000000000000000000000000000011",
        "to_address": "0xnorm000000000000000000000000000000000012",
        "value_eth": 2.5,
        "gas_price": 45.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_normal = agent_1_monitor(tx_normal)
    tx_normal = agent_2_behavioral(tx_normal)
    tx_normal = agent_3_network(tx_normal)
    tx_normal = agent_4_decision(tx_normal)
    print_result(tx_normal)
    
    if tx_normal.get('agent_2_score', 0) == 0:
        print("  ✅ Agent 2 correctly classified as low risk")
    else:
        print(f"  ⚠️  Agent 2 flagged normal transaction (score: {tx_normal.get('agent_2_score', 0)})")
    
    # Test 2.3: Mixer transaction (USE REAL MIXER)
    print("\n2.3 Mixer Transaction (Multi-Agent Detection):")
    tx_mixer = {
        "tx_hash": "0xtest_ml_mixer_001",
        "from_address": "0xnorm000000000000000000000000000000000013",
        "to_address": MIXER_WALLETS[1] if len(MIXER_WALLETS) > 1 else MIXER_WALLETS[0],
        "value_eth": 12.0,
        "gas_price": 60.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_mixer = agent_1_monitor(tx_mixer)
    tx_mixer = agent_2_behavioral(tx_mixer)
    tx_mixer = agent_3_network(tx_mixer)
    tx_mixer = agent_4_decision(tx_mixer)
    print_result(tx_mixer)
    
    a1_score = tx_mixer.get('agent_1_score', 0)
    a2_score = tx_mixer.get('agent_2_score', 0)
    if a1_score > 0 and a2_score > 0:
        print("  ✅ Both Agent 1 and Agent 2 detected risk")
    elif a1_score > 0:
        print("  ⚠️  Only Agent 1 detected risk (Agent 2 might need more mixer examples in training)")
    else:
        print("  ❌ Neither agent detected the mixer transaction")
    
    print("\n✓ Agent 2 Tests Complete")


def test_agent_3_network():
    """Test Agent 3: Network Analysis"""
    print_header("TEST 3: NETWORK ANALYSIS (Agent 3)")
    
    if not AGENT_3_AVAILABLE:
        print("⚠️  Agent 3 module not available. Skipping network tests.")
        return
    
    reset_agent_3_state()
    base_time = datetime.now(UTC)
    
    # Test 3.1: Circular transfer
    print("\n3.1 Circular Transfer Pattern (A -> B -> C -> A):")
    
    wallets = [
        "0xcirc000000000000000000000000000000000000",
        "0xcirc000000000000000000000000000000000001",
        "0xcirc000000000000000000000000000000000002",
    ]
    
    current_time = base_time
    
    # A -> B
    current_time += timedelta(minutes=5)
    tx1 = {
        "tx_hash": "0xtest_circ_001",
        "from_address": wallets[0],
        "to_address": wallets[1],
        "value_eth": 5.0,
        "gas_price": 50.0,
        "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
    }
    tx1 = agent_1_monitor(tx1)
    tx1 = agent_2_behavioral(tx1)
    tx1 = agent_3_network(tx1)
    tx1 = agent_4_decision(tx1)
    print(f"  Step 1 (A->B): Agent 3 Score = {tx1.get('agent_3_score', 0)}")
    
    # B -> C
    current_time += timedelta(minutes=5)
    tx2 = {
        "tx_hash": "0xtest_circ_002",
        "from_address": wallets[1],
        "to_address": wallets[2],
        "value_eth": 4.5,
        "gas_price": 50.0,
        "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
    }
    tx2 = agent_1_monitor(tx2)
    tx2 = agent_2_behavioral(tx2)
    tx2 = agent_3_network(tx2)
    tx2 = agent_4_decision(tx2)
    print(f"  Step 2 (B->C): Agent 3 Score = {tx2.get('agent_3_score', 0)}")
    
    # C -> A (Completes the circle)
    current_time += timedelta(minutes=5)
    tx3 = {
        "tx_hash": "0xtest_circ_003",
        "from_address": wallets[2],
        "to_address": wallets[0],
        "value_eth": 4.0,
        "gas_price": 50.0,
        "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
    }
    tx3 = agent_1_monitor(tx3)
    tx3 = agent_2_behavioral(tx3)
    tx3 = agent_3_network(tx3)
    tx3 = agent_4_decision(tx3)
    print_result(tx3, "\n  Step 3 - Completing Circle (C -> A):")
    
    if tx3.get('agent_3_score', 0) >= 40:
        print("\n✅ TEST PASSED: Circular pattern detected")
    else:
        print(f"\n❌ TEST FAILED: Should detect circular pattern (got score: {tx3.get('agent_3_score', 0)})")
    
    # Test 3.2: Fan-out (Smurfing)
    print("\n3.2 Fan-Out Pattern (Smurfing - 1 sender to many receivers):")
    reset_agent_3_state()
    
    sender = "0xsmurf00000000000000000000000000000000000"
    current_time = base_time
    
    for i in range(20):
        current_time += timedelta(seconds=30)
        tx_fanout = {
            "tx_hash": f"0xtest_fanout_{i:03d}",
            "from_address": sender,
            "to_address": f"0xrecv{i:037x}",
            "value_eth": 0.5,
            "gas_price": 50.0,
            "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
        }
        tx_fanout = agent_1_monitor(tx_fanout)
        tx_fanout = agent_2_behavioral(tx_fanout)
        tx_fanout = agent_3_network(tx_fanout)
        tx_fanout = agent_4_decision(tx_fanout)
        
        if i == 19:
            print_result(tx_fanout, f"\n  After {i+1} distributions:")
    
    if tx_fanout.get('agent_3_score', 0) >= 25:
        print("\n✅ TEST PASSED: Fan-out (smurfing) detected")
    else:
        print(f"\n❌ TEST FAILED: Should detect fan-out (got score: {tx_fanout.get('agent_3_score', 0)})")
    
    # Test 3.3: Fan-in (Collection)
    print("\n3.3 Fan-In Pattern (Collection - many senders to 1 receiver):")
    reset_agent_3_state()
    
    collector = "0xcollect0000000000000000000000000000000000"
    current_time = base_time
    
    for i in range(20):
        current_time += timedelta(seconds=30)
        tx_fanin = {
            "tx_hash": f"0xtest_fanin_{i:03d}",
            "from_address": f"0xsend{i:038x}",
            "to_address": collector,
            "value_eth": 0.3,
            "gas_price": 50.0,
            "timestamp": current_time.isoformat().replace('+00:00', 'Z'),
        }
        tx_fanin = agent_1_monitor(tx_fanin)
        tx_fanin = agent_2_behavioral(tx_fanin)
        tx_fanin = agent_3_network(tx_fanin)
        tx_fanin = agent_4_decision(tx_fanin)
        
        if i == 19:
            print_result(tx_fanin, f"\n  After {i+1} deposits:")
    
    if tx_fanin.get('agent_3_score', 0) >= 25:
        print("\n✅ TEST PASSED: Fan-in (collection) detected")
    else:
        print(f"\n❌ TEST FAILED: Should detect fan-in (got score: {tx_fanin.get('agent_3_score', 0)})")
    
    print("\n✓ Agent 3 Tests Complete")


def test_full_pipeline():
    """Test complete pipeline with all agents"""
    print_header("TEST 4: FULL PIPELINE INTEGRATION")
    
    if not MALICIOUS_WALLETS or not MIXER_WALLETS:
        print("⚠️  WARNING: No threat wallets loaded. Skipping integration tests.")
        return
    
    base_time = datetime.now(UTC)
    
    # Test 4.1: Maximum risk transaction (USE REAL WALLETS)
    print("\n4.1 Maximum Risk Transaction (Malicious -> Mixer, High Value):")
    tx_max_risk = {
        "tx_hash": "0xtest_maxrisk_001",
        "from_address": MALICIOUS_WALLETS[2] if len(MALICIOUS_WALLETS) > 2 else MALICIOUS_WALLETS[0],
        "to_address": MIXER_WALLETS[2] if len(MIXER_WALLETS) > 2 else MIXER_WALLETS[0],
        "value_eth": 120.0,
        "gas_price": 300.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    
    tx_max_risk = agent_1_monitor(tx_max_risk)
    tx_max_risk = agent_2_behavioral(tx_max_risk)
    tx_max_risk = agent_3_network(tx_max_risk)
    tx_max_risk = agent_4_decision(tx_max_risk)
    print_result(tx_max_risk)
    
    if tx_max_risk.get('final_status') == 'DENY':
        print("\n✅ TEST PASSED: High-risk transaction correctly denied")
    else:
        print(f"\n❌ TEST FAILED: Expected DENY, got {tx_max_risk.get('final_status')} (score: {tx_max_risk.get('final_score')})")
    
    # Test 4.2: Medium risk transaction
    print("\n4.2 Medium Risk Transaction (Should flag for review):")
    tx_med_risk = {
        "tx_hash": "0xtest_medrisk_001",
        "from_address": "0xnorm000000000000000000000000000000000020",
        "to_address": "0xnorm000000000000000000000000000000000021",
        "value_eth": 75.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    
    tx_med_risk = agent_1_monitor(tx_med_risk)
    tx_med_risk = agent_2_behavioral(tx_med_risk)
    tx_med_risk = agent_3_network(tx_med_risk)
    tx_med_risk = agent_4_decision(tx_med_risk)
    print_result(tx_med_risk)
    
    status = tx_med_risk.get('final_status')
    if status in ['FLAG_FOR_REVIEW', 'DENY']:
        print(f"\n✅ TEST PASSED: Medium risk correctly flagged as {status}")
    else:
        print(f"\n⚠️  Got {status} (score: {tx_med_risk.get('final_score')}) - Expected FLAG_FOR_REVIEW")
    
    print("\n✓ Full Pipeline Tests Complete")


def print_system_config():
    """Print current system configuration"""
    print_header("SYSTEM CONFIGURATION")
    
    thresholds = get_system_thresholds()
    
    print("\nDecision Thresholds:")
    decision_th = thresholds.get('agent_4_decision', {})
    print(f"  DENY: >= {decision_th.get('deny_threshold', 75)}")
    print(f"  FLAG_FOR_REVIEW: >= {decision_th.get('review_threshold', 45)}")
    print(f"  APPROVE: < {decision_th.get('review_threshold', 45)}")
    
    print("\nAgent Weights:")
    print(f"  Agent 1 (Threat Intel): 0.8x")
    print(f"  Agent 2 (ML Behavioral): 1.2x")
    print(f"  Agent 3 (Network): 0.5x")
    
    print("\nAgent 1 Thresholds:")
    a1_th = thresholds.get('agent_1_threat', {})
    print(f"  threat_list_match: {a1_th.get('threat_list_match', 40)} points")
    print(f"  mixer_match: {a1_th.get('mixer_match', 50)} points")
    
    print("\nAgent 2 Thresholds:")
    a2_th = thresholds.get('agent_2_behavioral_ml', {})
    print(f"  very_high_risk: {a2_th.get('very_high_risk', 60)} points")
    print(f"  high_risk: {a2_th.get('high_risk', 40)} points")
    print(f"  moderate_risk: {a2_th.get('moderate_risk', 25)} points")
    
    if AGENT_3_AVAILABLE:
        print("\nAgent 3 Thresholds:")
        print(f"  circular_pattern: 40 points")
        print(f"  high_fanout: 25 points")
        print(f"  high_fanin: 25 points")
    
    print("\nScoring System:")
    print(f"  Final Score = (A1 × 0.8) + (A2 × 1.2) + (A3 × 0.5)")
    print(f"  Maximum Score: 150 (capped)")


def main():
    """Run all tests"""
    print("="*70)
    print("      BLOCKCHAIN FRAUD DETECTION - COMPLETE AGENT TESTING SUITE")
    print("="*70)
    print("Testing all agents: Threat Intel, ML Behavioral, Network Analysis")
    
    if not MALICIOUS_WALLETS or not MIXER_WALLETS:
        print("\n❌ ERROR: Could not load threat wallets!")
        print("Please run: python initialize_and_train.py")
        return
    
    print_system_config()
    
    # Run all tests
    test_agent_1_threat_intel()
    test_agent_2_ml_behavioral()
    test_agent_3_network()
    test_full_pipeline()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()