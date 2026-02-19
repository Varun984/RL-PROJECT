"""Quick test for demo data generation"""
import sys
sys.path.insert(0, 'hrl_sarp')

from scripts.generate_demo_data import generate_market_data, generate_macro_data
from pathlib import Path
import shutil

# Clean up any existing test data
test_dir = Path("data/test")
if test_dir.exists():
    shutil.rmtree(test_dir)

print("Generating test data...")
generate_market_data("2023-01-01", "2023-01-31", test_dir)
generate_macro_data("2023-01-01", "2023-01-31", test_dir)

print("\n✅ Data generation successful!")
print(f"Check {test_dir} for generated files")

# List generated files
print("\nGenerated files:")
for file in test_dir.rglob("*.parquet"):
    print(f"  - {file}")
