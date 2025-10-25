"""
Agent Testing Suite for Blockchain Fraud Detection MVP
Tests each agent individually with calibrated thresholds
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
    print("[agents.py] Agent 3 (Network Analysis) loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import agent_3_network: {e}")
    AGENT_3_AVAILABLE = False
    def agent_3_network(transaction):
        transaction['agent_3_score'] = 0
        transaction['agent_3_reasons'] = ["Agent 3 skipped (module not found)"]
        return transaction


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
    
    # Combine all reasons
    reasons = []
    for key in ['agent_1_reasons', 'agent_2_reasons', 'agent_3_reasons']:
        agent_reasons = transaction.get(key, [])
        if isinstance(agent_reasons, list):
            reasons.extend(agent_reasons)
    
    if reasons:
        print(f"  Reasons:")
        for reason in reasons:
            print(f"    - {reason}")
    
    return transaction


def test_agent_1_threat_intel():
    """Test Agent 1: Threat Intelligence"""
    print_header("TEST 1: THREAT INTELLIGENCE (Agent 1)")
    
    # Load threat list
    threat_file = PROJECT_ROOT / 'data' / 'dark_web_wallets.txt'
    if threat_file.exists():
        with open(threat_file, 'r') as f:
            threat_count = len([line for line in f if line.strip()])
        print(f"Loaded threat list: {threat_count} addresses.\n")
    
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
    
    # Test 1.2: Malicious sender
    tx_malicious_sender = {
        "tx_hash": "0xtest_malicious_002",
        "from_address": "0xbad0000000000000000000000000000000000000",
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
    
    # Test 1.3: Mixer transaction
    tx_mixer = {
        "tx_hash": "0xtest_mixer_003",
        "from_address": "0xnorm000000000000000000000000000000000003",
        "to_address": "0xmix0000000000000000000000000000000000000",
        "value_eth": 8.0,
        "gas_price": 50.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    tx_mixer = agent_1_monitor(tx_mixer)
    tx_mixer = agent_2_behavioral(tx_mixer)
    tx_mixer = agent_3_network(tx_mixer)
    tx_mixer = agent_4_decision(tx_mixer)
    print_result(tx_mixer, "1.3 Mixer Transaction:")
    
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
    
    print("\n✓ Agent 1 Tests Complete")


def test_agent_3_network():
    """Test Agent 3: Network Analysis"""
    print_header("TEST 2: NETWORK ANALYSIS (Agent 3)")
    
    if not AGENT_3_AVAILABLE:
        print("⚠️  Agent 3 module not available. Skipping network tests.")
        return
    
    reset_agent_3_state()
    base_time = datetime.now(UTC)
    
    # Test 2.1: Circular transfer (A -> B -> C -> A)
    print("\n2.1 Circular Transfer Pattern (A -> B -> C -> A):")
    
    wallets = [
        "0xcirc000000000000000000000000000000000000",
        "0xcirc000000000000000000000000000000000001",
        "0xcirc000000000000000000000000000000000002",
    ]
    
    current_time = base_time
    transactions = []
    
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
    transactions.append(tx1)
    
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
    transactions.append(tx2)
    
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
    print_result(tx3, "\n  Completing Circle (C -> A):")
    
    # Check if circular pattern was detected
    if tx3.get('agent_3_score', 0) > 0:
        print("\n✓ TEST PASSED: Circular pattern detected")
    else:
        print("\n❌ TEST FAILED: Should detect circular pattern")


def test_full_pipeline():
    """Test complete pipeline with all agents"""
    print_header("TEST 3: FULL PIPELINE INTEGRATION")
    
    base_time = datetime.now(UTC)
    
    # High-risk transaction
    tx_high_risk = {
        "tx_hash": "0xtest_highrisk_001",
        "from_address": "0xbad0000000000000000000000000000000000000",
        "to_address": "0xmix0000000000000000000000000000000000000",
        "value_eth": 75.0,
        "gas_price": 250.0,
        "timestamp": base_time.isoformat().replace('+00:00', 'Z'),
    }
    
    print("\n3.1 High-Risk Transaction (Malicious -> Mixer):")
    tx_high_risk = agent_1_monitor(tx_high_risk)
    tx_high_risk = agent_2_behavioral(tx_high_risk)
    tx_high_risk = agent_3_network(tx_high_risk)
    tx_high_risk = agent_4_decision(tx_high_risk)
    print_result(tx_high_risk)
    
    if tx_high_risk.get('final_status') == 'DENY':
        print("\n✓ TEST PASSED: High-risk transaction correctly denied")
    else:
        print(f"\n❌ TEST FAILED: Expected DENY, got {tx_high_risk.get('final_status')}")


def print_system_config():
    """Print current system configuration"""
    print_header("SYSTEM CONFIGURATION")
    
    thresholds = get_system_thresholds()
    
    print("\nDecision Thresholds:")
    decision_th = thresholds.get('agent_4_decision', {})
    print(f"  DENY: >= {decision_th.get('deny_threshold', 80)}")
    print(f"  FLAG_FOR_REVIEW: >= {decision_th.get('review_threshold', 45)}")
    print(f"  APPROVE: < {decision_th.get('approve_threshold', 0)}")
    
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
    print("           BLOCKCHAIN FRAUD DETECTION - AGENT TESTING SUITE")
    print("="*70)
    print("Testing calibrated multi-agent system")
    
    print_system_config()
    
    # Run tests
    test_agent_1_threat_intel()
    test_agent_3_network()
    test_full_pipeline()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()