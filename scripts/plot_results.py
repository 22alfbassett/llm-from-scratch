import argparse
import glob
import json
import os
import time

import matplotlib.pyplot as plt


def plot_all_runs(dataset=None, top_n=100):
    results_dir = os.path.join(os.getcwd(), "results")
    if dataset:
        pattern = os.path.join(results_dir, f"{dataset}_*.json")
    else:
        pattern = os.path.join(results_dir, "*.json")

    files = glob.glob(pattern)
    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    all_data = []
    for f in files:
        try:
            with open(f) as res:
                data = json.load(res)
                if not data.get("loss"):
                    continue
                tail = max(1, len(data["loss"]) // 20)
                final_loss = sum(data["loss"][-tail:]) / tail
                all_data.append(
                    {
                        "data": data,
                        "final_loss": final_loss,
                        "params": data.get("params", {}),
                    }
                )
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    if not all_data:
        print("No valid data found to plot.")
        return

    all_data.sort(key=lambda x: x["final_loss"])
    print(f"\n--- ranking (top {min(len(all_data), 20)}) ---")
    header = f"{'loss':<8} | {'L/H/E':<10} | {'act/norm':<15} | {'bs':<2} | {'lr':<6} | {'dtype':<7} | {'tie':<3}"
    print(header)
    print("-" * len(header))
    for e in all_data[:20]:
        p = e["params"]
        cfg, an = (
            f"{p.get('n_layer')}/{p.get('n_head')}/{p.get('n_embd')}",
            f"{p.get('act')}/{p.get('norm')}",
        )
        print(
            f"{e['final_loss']:<8.4f} | {cfg:<10} | {an:<15} | {p.get('batch_size'):<2} | {p.get('lr'):<6} | {p.get('dtype', 'f32'):<7} | {str(p.get('tie_weights'))[:3]}"
        )

    plt.figure(figsize=(16, 10))
    for e in all_data[:top_n]:
        p, loss = e["params"], e["data"]["loss"]
        label = f"L{p.get('n_layer')} H{p.get('n_head')} E{p.get('n_embd')} {p.get('act')}/{p.get('norm')} bs{p.get('batch_size')} lr{p.get('lr')} {p.get('dtype', 'f32')} -> {e['final_loss']:.3f}"
        if len(loss) > 100:
            w = max(1, len(loss) // 50)
            loss = [sum(loss[i : i + w]) / w for i in range(len(loss) - w)]
        plt.plot(loss, label=label, alpha=0.8, linewidth=1.5)

    plt.title(f"comparison: {len(all_data[:top_n])} runs")
    plt.xlabel("steps (smoothed)")
    plt.ylabel("loss")
    plt.legend(
        bbox_to_anchor=(1.0, -0.15),
        loc="upper right",
        fontsize="x-small",
        ncol=3,
        frameon=True,
    )
    plt.grid(True, alpha=0.2)
    filename = f"comparison_{dataset}_{int(time.time())}.png" if dataset else "comparison.png"
    out = os.path.join(results_dir, filename)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str)
    parser.add_argument("--top", "-t", type=int, default=100)
    args = parser.parse_args()
    plot_all_runs(args.dataset, top_n=args.top)
