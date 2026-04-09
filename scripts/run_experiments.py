import argparse
import itertools
import subprocess
import sys


def run_experiments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # experiments grid
    grid = {
        "act": ["swiglu", "gelu"],
        "norm": ["rmsnorm", "layernorm"],
        "n_layer": [8, 16],
        "n_embd": [256, 512],
        "batch_size": [8, 16],
        "lr": [1e-3, 5e-4],
        "steps": [200 if args.quick else 2000],
        "vocab_size": [750],
        "dtype": ["float16"],
    }

    keys, values = list(grid.keys()), list(grid.values())
    combinations = list(itertools.product(*values))

    print(f"--- starting {len(combinations)} experiments: {args.dataset} ---")
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        # All arguments after 'train' subcommand as they are shared/added there too
        cmd = [sys.executable, "-m", "llm.cli", "train", "--dataset", args.dataset]
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        print(f"\n[{i + 1}/{len(combinations)}] params: {params}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Experiment failed: {e}")
            continue

    print("\n--- experiments complete ---")
    subprocess.run([sys.executable, "scripts/plot_results.py", "--dataset", args.dataset])


if __name__ == "__main__":
    run_experiments()
