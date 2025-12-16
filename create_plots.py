import os
import matplotlib.pyplot as plt
import ab.nn.api as api

OUTDIR = "docs/figures"
os.makedirs(OUTDIR, exist_ok=True)

df = api.data()
df["prm_str"] = df["prm"].astype(str)
run_cols = ["task", "dataset", "metric", "nn", "prm_str", "transform_code"]

idx = df.groupby(run_cols)["accuracy"].idxmax().dropna().astype(int)
best = df.loc[idx].copy().rename(columns={"accuracy": "best_accuracy", "epoch": "best_epoch"})

task_stats = best.groupby("task")["best_accuracy"].mean().sort_values(ascending=False)

plt.figure()
task_stats.plot(kind="bar")
plt.ylabel("Mean best accuracy")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/per_task_mean_best_accuracy.png", dpi=200)
plt.close()
