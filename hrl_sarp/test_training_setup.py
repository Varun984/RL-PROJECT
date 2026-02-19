"""
Quick test script to verify training setup works.
Run this before attempting full training.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_data_loader():
    """Test that data loader can load training data."""
    print("=" * 60)
    print("Testing Data Loader")
    print("=" * 60)
    
    try:
        from data.training_data_loader import TrainingDataLoader
        
        loader = TrainingDataLoader(config_path="config/data_config.yaml")
        print("✓ TrainingDataLoader initialized")
        
        # Test macro data loading
        print("\nLoading macro training data (2020-2021)...")
        macro_data = loader.load_macro_training_data("2020-01-01", "2021-12-31")
        
        print(f"✓ Macro data loaded:")
        print(f"  - macro_states: {macro_data['macro_states'].shape}")
        print(f"  - sector_embeddings: {macro_data['sector_embeddings'].shape}")
        print(f"  - sector_returns: {macro_data['sector_returns'].shape}")
        print(f"  - regime_labels: {macro_data['regime_labels'].shape}")
        
        # Test micro data loading
        print("\nLoading micro training data (2020-2021)...")
        micro_data = loader.load_micro_training_data("2020-01-01", "2021-12-31", max_stocks=50)
        
        print(f"✓ Micro data loaded:")
        print(f"  - stock_returns: {micro_data['stock_returns'].shape}")
        print(f"  - stock_features: {micro_data['stock_features'].shape}")
        print(f"  - stock_to_sector: {micro_data['stock_to_sector'].shape}")
        print(f"  - stock_masks: {micro_data['stock_masks'].shape}")
        
        print("\n✅ Data loader test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Data loader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_initialization():
    """Test that agents can be initialized."""
    print("\n" + "=" * 60)
    print("Testing Agent Initialization")
    print("=" * 60)
    
    try:
        import torch
        from agents.macro_agent import MacroAgent
        from agents.micro_agent import MicroAgent
        
        device = torch.device("cpu")
        
        # Test Macro agent
        print("\nInitializing MacroAgent...")
        macro_agent = MacroAgent(
            config_path="config/macro_agent_config.yaml",
            device=device,
        )
        print("✓ MacroAgent initialized")
        
        # Test Micro agent
        print("\nInitializing MicroAgent...")
        micro_agent = MicroAgent(
            config_path="config/micro_agent_config.yaml",
            device=device,
        )
        print("✓ MicroAgent initialized")
        
        print("\n✅ Agent initialization test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Agent initialization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_imports():
    """Test that training modules can be imported."""
    print("\n" + "=" * 60)
    print("Testing Training Module Imports")
    print("=" * 60)
    
    try:
        from training.pretrain_macro import pretrain_macro
        print("✓ pretrain_macro imported")
        
        from training.pretrain_micro import pretrain_micro
        print("✓ pretrain_micro imported")
        
        from training.trainer_utils import set_global_seed
        print("✓ trainer_utils imported")
        
        print("\n✅ Training imports test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training imports test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HRL-SARP TRAINING SETUP TEST")
    print("=" * 60)
    print("\nThis script tests if training infrastructure is ready.")
    print("It will check:")
    print("  1. Data loader can load training data")
    print("  2. Agents can be initialized")
    print("  3. Training modules can be imported")
    print()
    
    results = []
    
    # Run tests
    results.append(("Training Imports", test_training_imports()))
    results.append(("Agent Initialization", test_agent_initialization()))
    results.append(("Data Loader", test_data_loader()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run training.")
        print("\nNext steps:")
        print("  1. Run Phase 1: python main.py train --phase 1")
        print("  2. Run Phase 2: python main.py train --phase 2")
        print("  3. Run all phases: python main.py train")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
