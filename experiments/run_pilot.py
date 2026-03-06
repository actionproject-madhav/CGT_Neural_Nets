"""
Master script for the CGT Neural Net Pilot Study.

Runs the entire experiment end-to-end:
  1. Data generation
  2. MLP baseline training (multi-seed)
  3. DeepSets baseline training (multi-seed)
  4. CGT-Net training (multi-seed)
  5. DQN baseline training (multi-seed)
  6. Evaluation across all test sets
  7. Results saving

Usage:
    python -m experiments.run_pilot
    python -m experiments.run_pilot --seeds 42 123 --dqn-episodes 10000
"""

from __future__ import annotations
import os
import sys
import json
import time
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generator import NimDataGenerator
from models.mlp_baseline import build_mlp_model
from models.deepsets_baseline import build_deepsets_model
from models.cgt_net import build_cgt_model
from models.dqn_baseline import DQNAgent
from training.trainer import SupervisedTrainer
from training.rl_trainer import DQNTrainer
from evaluation.evaluator import PilotEvaluator


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(
            os.path.dirname(__file__), "configs", "pilot_config.yaml"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def run_pilot(args):
    config = load_config(args.config)
    seeds = args.seeds or config["experiment"]["seeds"]
    results_dir = config["experiment"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    max_pad = config["data"]["max_pad_heaps"]
    dqn_episodes = args.dqn_episodes or config["models"]["dqn"]["n_episodes"]

    start_time = time.time()
    all_results = {}

    # ===== Step 1: Data Generation =====
    print("\n" + "=" * 60)
    print("STEP 1: Generating Datasets")
    print("=" * 60)

    generator = NimDataGenerator(max_pad_heaps=max_pad, seed=42)
    datasets = generator.generate_all_datasets(
        train_val_test_split=tuple(config["data"]["train_val_test_split"])
    )
    print(generator.get_dataset_stats(datasets))

    test_datasets = {
        k: v for k, v in datasets.items() if k.startswith("test")
    }

    evaluator = PilotEvaluator(results_dir=results_dir)

    # ===== Step 2: MLP Baseline =====
    print("\n" + "=" * 60)
    print("STEP 2: Training MLP Baseline")
    print("=" * 60)

    mlp_cfg = config["models"]["mlp"]
    mlp_trainer = SupervisedTrainer(
        model_builder=lambda: build_mlp_model(
            max_heaps=max_pad,
            hidden_units=mlp_cfg["hidden_units"],
            n_hidden=mlp_cfg["n_hidden"],
            dropout_rate=mlp_cfg["dropout_rate"],
        ),
        model_name="mlp_baseline",
        results_dir=results_dir,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
    )
    mlp_results = mlp_trainer.train_multi_seed(
        datasets["train"], datasets["val"], seeds
    )

    mlp_eval_all = []
    mlp_move_rates = []
    mlp_thresholds = []
    for r in mlp_results:
        ev = evaluator.evaluate_supervised_model(r["model"], "mlp_baseline", test_datasets)
        mlp_eval_all.append(ev)
        mr = evaluator.compute_optimal_move_rate(
            r["model"], "mlp_baseline", datasets["test_id"], max_pad
        )
        mlp_move_rates.append(mr)
        mlp_thresholds.append(evaluator.samples_to_threshold(r["history"]))

    all_results["mlp_baseline"] = evaluator.aggregate_multi_seed_results(mlp_eval_all)
    all_results["mlp_baseline"]["optimal_move_rate"] = {
        "mean": float(np.mean(mlp_move_rates)),
        "std": float(np.std(mlp_move_rates)),
    }
    all_results["mlp_baseline"]["samples_to_95"] = [t for t in mlp_thresholds]

    # ===== Step 3: DeepSets Baseline =====
    print("\n" + "=" * 60)
    print("STEP 3: Training DeepSets Baseline")
    print("=" * 60)

    ds_cfg = config["models"]["deepsets"]
    ds_trainer = SupervisedTrainer(
        model_builder=lambda: build_deepsets_model(
            max_heaps=max_pad,
            phi_units=ds_cfg["phi_units"],
            phi_layers=ds_cfg["phi_layers"],
            rho_units=ds_cfg["rho_units"],
            rho_layers=ds_cfg["rho_layers"],
            dropout_rate=ds_cfg["dropout_rate"],
        ),
        model_name="deepsets_baseline",
        results_dir=results_dir,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
    )
    ds_results = ds_trainer.train_multi_seed(
        datasets["train"], datasets["val"], seeds
    )

    ds_eval_all = []
    ds_move_rates = []
    ds_thresholds = []
    for r in ds_results:
        ev = evaluator.evaluate_supervised_model(
            r["model"], "deepsets_baseline", test_datasets
        )
        ds_eval_all.append(ev)
        mr = evaluator.compute_optimal_move_rate(
            r["model"], "deepsets_baseline", datasets["test_id"], max_pad
        )
        ds_move_rates.append(mr)
        ds_thresholds.append(evaluator.samples_to_threshold(r["history"]))

    all_results["deepsets_baseline"] = evaluator.aggregate_multi_seed_results(ds_eval_all)
    all_results["deepsets_baseline"]["optimal_move_rate"] = {
        "mean": float(np.mean(ds_move_rates)),
        "std": float(np.std(ds_move_rates)),
    }
    all_results["deepsets_baseline"]["samples_to_95"] = [t for t in ds_thresholds]

    # ===== Step 4: CGT-Net =====
    print("\n" + "=" * 60)
    print("STEP 4: Training CGT-Decomposition Network")
    print("=" * 60)

    cgt_cfg = config["models"]["cgt_net"]
    cgt_trainer = SupervisedTrainer(
        model_builder=lambda: build_cgt_model(
            max_heaps=max_pad,
            embed_dim=cgt_cfg["embed_dim"],
            encoder_units=cgt_cfg["encoder_units"],
            encoder_layers=cgt_cfg["encoder_layers"],
            max_grundy=cgt_cfg["max_grundy"],
            dropout_rate=cgt_cfg["dropout_rate"],
        ),
        model_name="cgt_net",
        results_dir=results_dir,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["patience"],
    )
    cgt_results = cgt_trainer.train_multi_seed(
        datasets["train"], datasets["val"], seeds
    )

    cgt_eval_all = []
    cgt_move_rates = []
    cgt_thresholds = []
    for r in cgt_results:
        ev = evaluator.evaluate_supervised_model(r["model"], "cgt_net", test_datasets)
        cgt_eval_all.append(ev)
        mr = evaluator.compute_optimal_move_rate(
            r["model"], "cgt_net", datasets["test_id"], max_pad
        )
        cgt_move_rates.append(mr)
        cgt_thresholds.append(evaluator.samples_to_threshold(r["history"]))

    all_results["cgt_net"] = evaluator.aggregate_multi_seed_results(cgt_eval_all)
    all_results["cgt_net"]["optimal_move_rate"] = {
        "mean": float(np.mean(cgt_move_rates)),
        "std": float(np.std(cgt_move_rates)),
    }
    all_results["cgt_net"]["samples_to_95"] = [t for t in cgt_thresholds]

    # ===== Step 5: DQN Baseline =====
    print("\n" + "=" * 60)
    print("STEP 5: Training DQN Baseline")
    print("=" * 60)

    dqn_trainer = DQNTrainer(
        max_heaps=max_pad,
        max_heap_size=15,
        results_dir=results_dir,
    )
    dqn_results = dqn_trainer.train_multi_seed(
        seeds=seeds,
        n_episodes=dqn_episodes,
        heap_counts=config["data"]["train_heap_counts"],
        heap_size_range=tuple(config["data"]["train_heap_range"]),
    )

    dqn_eval_all = []
    for r in dqn_results:
        ev = evaluator.evaluate_dqn_agent(r["agent"], test_datasets)
        dqn_eval_all.append(ev)

    all_results["dqn"] = evaluator.aggregate_multi_seed_results(dqn_eval_all)

    # ===== Step 6: Save & Display Results =====
    print("\n" + "=" * 60)
    print("STEP 6: Final Results")
    print("=" * 60)

    evaluator.print_comparison_table(all_results)
    evaluator.save_results(all_results)

    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed / 60:.1f} minutes")

    # Save training histories for plotting
    histories = {
        "mlp_baseline": [r["history"] for r in mlp_results],
        "deepsets_baseline": [r["history"] for r in ds_results],
        "cgt_net": [r["history"] for r in cgt_results],
        "dqn": [r["history"] for r in dqn_results],
    }
    with open(os.path.join(results_dir, "training_histories.json"), "w") as f:
        json.dump(histories, f)

    print("\nPilot study complete. Run 'python -m analysis.plot_results' to generate plots.")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CGT Neural Net Pilot Study")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config YAML"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None, help="Random seeds"
    )
    parser.add_argument(
        "--dqn-episodes",
        type=int,
        default=None,
        help="Number of DQN training episodes",
    )
    args = parser.parse_args()
    run_pilot(args)


if __name__ == "__main__":
    main()
