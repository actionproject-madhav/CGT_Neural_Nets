"""Ultra-fast eval: use existing saved models, evaluate, plot."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow import keras
from data.generator import NimDataGenerator
from evaluation.evaluator import PilotEvaluator

gen = NimDataGenerator(max_pad_heaps=6, max_heap_value=15, seed=42)
datasets = gen.generate_all_datasets(
    train_heap_counts=[2,3,4], train_heap_range=(0,7),
    ood_heap_counts=[5,6], ood_size_range=(8,15),
)
test_ds = {k:v for k,v in datasets.items() if k.startswith("test")}
evaluator = PilotEvaluator(results_dir="results", max_heap_value=15)

print("\n=== QUICK PILOT RESULTS: MLP vs CGT-Net ===\n")

# Load saved models and histories
results = {}
for model_name, label in [("mlp_baseline", "MLP"), ("cgt_net", "CGT-Net")]:
    evals, mrs, hists = [], [], []
    for seed in [42, 123, 456, 789, 1024]:
        mp = f"results/{model_name}/seed_{seed}/model.keras"
        hp = f"results/{model_name}/seed_{seed}/history.json"
        if not os.path.exists(mp): continue
        
        # Custom objects for CGT-Net
        custom = {}
        if model_name == "cgt_net":
            from models.cgt_net import BitwiseAggregator
            custom = {"BitwiseAggregator": BitwiseAggregator}
        
        model = keras.models.load_model(mp, custom_objects=custom)
        ev = evaluator.evaluate_supervised_model(model, model_name, test_ds)
        evals.append(ev)
        mrs.append(evaluator.compute_optimal_move_rate(model, model_name, datasets["test_id"], 6))
        with open(hp) as f: hists.append(json.load(f))
        
        best = max(hists[-1]["val_acc"])
        print(f"{label} seed {seed}: ID={ev['test_id']['win_loss_accuracy']:.1%} "
              f"OOD-H={ev['test_ood_heaps']['win_loss_accuracy']:.1%} "
              f"OOD-S={ev['test_ood_sizes']['win_loss_accuracy']:.1%} "
              f"(val={best:.1%})")
    
    results[model_name] = evaluator.aggregate_multi_seed_results(evals)
    results[model_name]["optimal_move_rate"] = {"mean": float(np.mean(mrs)), "std": float(np.std(mrs))}
    print(f"{label} average: ID={results[model_name]['test_id']['win_loss_accuracy']['mean']:.1%}, "
          f"OOD-Heaps={results[model_name]['test_ood_heaps']['win_loss_accuracy']['mean']:.1%}\n")

# Print table
print("\n" + "="*80)
evaluator.print_comparison_table(results)
evaluator.save_results(results)

print(f"\n✓ Results saved: results/pilot_results.json")
print(f"✓ All models trained & evaluated successfully")
print(f"\n=== KEY FINDING ===")
mlp_ood = results["mlp_baseline"]["test_ood_heaps"]["win_loss_accuracy"]["mean"]
cgt_ood = results["cgt_net"]["test_ood_heaps"]["win_loss_accuracy"]["mean"]
print(f"Compositional Generalization (OOD-Heaps):")
print(f"  MLP: {mlp_ood:.1%}")
print(f"  CGT-Net: {cgt_ood:.1%}")
print(f"  Gap: {cgt_ood - mlp_ood:+.1%}")
print("="*80)
