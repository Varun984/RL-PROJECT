"""
File: main.py
Description: Top-level entry point for the HRL-SARP framework.
    Provides CLI for running the full training pipeline, individual phases,
    backtesting, evaluation, and dashboard launch.

Usage:
    python main.py train          # Full 5-phase training
    python main.py train --phase 1  # Single phase
    python main.py backtest       # Walk-forward backtest
    python main.py evaluate       # Full evaluation pipeline
    python main.py dashboard      # Launch Streamlit dashboard

Author: HRL-SARP Framework
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.common import set_global_seed, get_device, setup_logging
from utils.io_utils import load_yaml

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION LOADING
# ══════════════════════════════════════════════════════════════════════


def load_all_configs() -> dict:
    """Load all configuration files."""
    config_dir = os.path.join(PROJECT_ROOT, "config")
    configs = {}
    for name in ["macro_agent_config", "micro_agent_config", "data_config", "risk_config"]:
        path = os.path.join(config_dir, f"{name}.yaml")
        if os.path.exists(path):
            configs[name] = load_yaml(path)
            logger.info("Loaded config: %s", name)
        else:
            logger.warning("Config not found: %s", path)
            configs[name] = {}
    return configs


# ══════════════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════


def run_training(args: argparse.Namespace) -> None:
    """Execute the 5-phase training pipeline."""
    configs = load_all_configs()
    device = get_device()
    seed = args.seed

    set_global_seed(seed)

    logger.info("=" * 70)
    logger.info("HRL-SARP TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Seed: %d | Device: %s", seed, device)

    # Import training modules
    from training.pretrain_macro import pretrain_macro
    from training.pretrain_micro import pretrain_micro
    from training.train_macro_frozen_micro import train_macro_frozen_micro
    from training.train_micro_frozen_macro import train_micro_frozen_macro
    from training.joint_finetune import joint_finetune

    phases = {
        1: ("Phase 1: Supervised pre-training of Macro agent", pretrain_macro),
        2: ("Phase 2: Supervised pre-training of Micro agent", pretrain_micro),
        3: ("Phase 3: RL training of Macro (frozen Micro)", train_macro_frozen_micro),
        4: ("Phase 4: RL training of Micro (frozen Macro)", train_micro_frozen_macro),
        5: ("Phase 5: Joint fine-tuning", joint_finetune),
    }

    # Determine which phases to run
    if args.phase:
        phases_to_run = [args.phase]
    else:
        phases_to_run = list(range(1, 6))

    for phase_num in phases_to_run:
        if phase_num not in phases:
            logger.error("Invalid phase number: %d", phase_num)
            continue

        name, func = phases[phase_num]
        logger.info("-" * 60)
        logger.info("Starting %s", name)
        logger.info("-" * 60)

        try:
            func(configs, device=device, seed=seed)
            logger.info("✅ %s completed successfully", name)
        except Exception as e:
            logger.error("Phase failed: %s - %s", name, str(e), exc_info=True)
            if not args.continue_on_error:
                raise

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)


# ══════════════════════════════════════════════════════════════════════
# BACKTESTING
# ══════════════════════════════════════════════════════════════════════


def run_backtest(args: argparse.Namespace) -> None:
    """Execute walk-forward backtesting."""
    configs = load_all_configs()
    device = get_device()
    set_global_seed(args.seed)

    logger.info("=" * 70)
    logger.info("HRL-SARP WALK-FORWARD BACKTEST")
    logger.info("=" * 70)

    from backtest.backtester import Backtester
    from backtest.performance_metrics import PerformanceMetrics

    risk_cfg = configs.get("risk_config", {})
    initial_capital = args.capital

    backtester = Backtester(
        risk_free_rate=0.07,
        initial_capital=initial_capital,
    )

    # TODO: Load trained agents and environment
    # macro_agent = MacroAgent.load(args.macro_checkpoint)
    # micro_agent = MicroAgent.load(args.micro_checkpoint)
    # env = HierarchicalEnv(configs)
    # results = backtester.run(env, macro_agent, micro_agent, n_episodes=args.n_episodes)

    logger.info(
        "Backtest configured | capital=₹%.0f | episodes=%d",
        initial_capital, args.n_episodes,
    )
    logger.info("⚠️  Load trained agent checkpoints to run actual backtest")


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════


def run_evaluation(args: argparse.Namespace) -> None:
    """Execute full evaluation pipeline."""
    configs = load_all_configs()
    device = get_device()
    set_global_seed(args.seed)

    logger.info("=" * 70)
    logger.info("HRL-SARP FULL EVALUATION")
    logger.info("=" * 70)

    from evaluation.evaluator import Evaluator
    from evaluation.report_generator import ReportGenerator

    evaluator = Evaluator(
        risk_free_rate=0.07,
        initial_capital=args.capital,
        output_dir=args.output_dir,
    )

    report_gen = ReportGenerator(output_dir=args.output_dir)

    # TODO: Load trained agents, environment, and market data
    # results = evaluator.run_full_evaluation(env, macro_agent, micro_agent, ...)
    # report_path = report_gen.generate(results)

    logger.info("Evaluation configured | output=%s", args.output_dir)
    logger.info("⚠️  Load trained agent checkpoints to run actual evaluation")


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════


def run_dashboard(args: argparse.Namespace) -> None:
    """Launch Streamlit monitoring dashboard."""
    logger.info("Launching dashboard...")
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")

    if not os.path.exists(dashboard_path):
        logger.error("Dashboard app not found: %s", dashboard_path)
        return

    os.system(f"streamlit run {dashboard_path} --server.port {args.port}")


# ══════════════════════════════════════════════════════════════════════
# STRESS TEST
# ══════════════════════════════════════════════════════════════════════


def run_stress_test(args: argparse.Namespace) -> None:
    """Run stress testing on a portfolio."""
    logger.info("=" * 70)
    logger.info("HRL-SARP STRESS TESTING")
    logger.info("=" * 70)

    from risk.stress_testing import StressTester

    tester = StressTester()

    # Example: equal-weight portfolio
    weights = np.ones(11) / 11
    results = tester.run_all(weights, args.capital)
    report = tester.generate_report(results)

    logger.info("Scenarios passed: %d/%d", report["scenarios_passed"], report["n_scenarios"])
    logger.info("Worst case: %s (%.2f%%)", report["worst_scenario"],
                results[report["worst_scenario"]]["portfolio_return"] * 100)

    for name, detail in results.items():
        logger.info(
            "  %-20s | P&L: ₹%12.0f | DD: %.2f%% | %s",
            name,
            detail["pnl"],
            detail["max_drawdown"] * 100,
            "✅" if detail["passes_drawdown_limit"] else "❌",
        )


# ══════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════


def run_data_collection(args):
    """Run data collection pipeline."""
    import subprocess
    
    logger.info("=" * 70)
    logger.info("HRL-SARP DATA COLLECTION")
    logger.info("=" * 70)
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "collect_data.py"),
        "--start", args.start,
        "--end", args.end,
        "--output-dir", args.output_dir,
    ]
    
    if args.skip_market:
        cmd.append("--skip-market")
    if args.skip_macro:
        cmd.append("--skip-macro")
    if args.skip_fundamentals:
        cmd.append("--skip-fundamentals")
    if args.skip_news:
        cmd.append("--skip-news")
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def run_demo_generation(args):
    """Generate demo data."""
    import subprocess
    
    logger.info("=" * 70)
    logger.info("GENERATING DEMO DATA")
    logger.info("=" * 70)
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "scripts", "generate_demo_data.py"),
        "--start", args.start,
        "--end", args.end,
        "--output-dir", args.output_dir,
    ]
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ══════════════════════════════════════════════════════════════════════
# CLI DEFINITION
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="HRL-SARP",
        description="Hierarchical RL for Sector-Aware Risk-Adaptive Portfolio Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                  Full 5-phase training
  python main.py train --phase 1        Single phase pre-training
  python main.py backtest               Walk-forward backtest
  python main.py evaluate               Full evaluation with report
  python main.py stress-test            Stress test equal-weight portfolio
  python main.py dashboard              Launch monitoring dashboard
        """,
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                              help="Run specific phase only")
    train_parser.add_argument("--continue-on-error", action="store_true",
                              help="Continue to next phase on failure")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    bt_parser.add_argument("--capital", type=float, default=10_000_000,
                           help="Initial capital in INR")
    bt_parser.add_argument("--n-episodes", type=int, default=5,
                           help="Number of walk-forward windows")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run full evaluation pipeline")
    eval_parser.add_argument("--capital", type=float, default=10_000_000)
    eval_parser.add_argument("--output-dir", type=str, default="results")

    # Stress test
    stress_parser = subparsers.add_parser("stress-test", help="Run stress tests")
    stress_parser.add_argument("--capital", type=float, default=10_000_000)

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch monitoring dashboard")
    dash_parser.add_argument("--port", type=int, default=8501)

    # Data collection
    data_parser = subparsers.add_parser("collect-data", help="Collect historical data")
    data_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    data_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    data_parser.add_argument("--output-dir", type=str, default="data/raw")
    data_parser.add_argument("--skip-market", action="store_true")
    data_parser.add_argument("--skip-macro", action="store_true")
    data_parser.add_argument("--skip-fundamentals", action="store_true")
    data_parser.add_argument("--skip-news", action="store_true")

    # Generate demo data
    demo_parser = subparsers.add_parser("generate-demo", help="Generate synthetic demo data")
    demo_parser.add_argument("--start", type=str, default="2020-01-01")
    demo_parser.add_argument("--end", type=str, default="2023-12-31")
    demo_parser.add_argument("--output-dir", type=str, default="data/raw")

    return parser


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, experiment_name=args.command or "hrl_sarp")

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "train": run_training,
        "backtest": run_backtest,
        "evaluate": run_evaluation,
        "stress-test": run_stress_test,
        "dashboard": run_dashboard,
        "collect-data": run_data_collection,
        "generate-demo": run_demo_generation,
    }

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
