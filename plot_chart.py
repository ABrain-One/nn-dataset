import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Data from your single training phase (LR = 1e-4) ---
log_data = """
Initialize training with lr: 0.0001, version: 0.3805808385281112, momentum: 0.2080728589260823, batch: 8, transform: norm_256
epoch 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:32<00:00, 16.62it/s]
Trial (accuracy 0.2581829999406761) for (txt-image, coco, clip, ConditionalVAE3, 1) saved
epoch 2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:31<00:00, 16.71it/s]
Trial (accuracy 0.2652767051625474) for (txt-image, coco, clip, ConditionalVAE3, 2) saved
epoch 3
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.84it/s]
Trial (accuracy 0.26652437014000435) for (txt-image, coco, clip, ConditionalVAE3, 3) saved
epoch 4
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.07it/s]
Trial (accuracy 0.2712622053199839) for (txt-image, coco, clip, ConditionalVAE3, 4) saved
epoch 5
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.05it/s]
Trial (accuracy 0.2698451138790523) for (txt-image, coco, clip, ConditionalVAE3, 5) saved
epoch 6
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.05it/s]
Trial (accuracy 0.270851590450679) for (txt-image, coco, clip, ConditionalVAE3, 6) saved
epoch 7
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.2714149546400409) for (txt-image, coco, clip, ConditionalVAE3, 7) saved
epoch 8
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:32<00:00, 16.51it/s]
Trial (accuracy 0.2698484012211595) for (txt-image, coco, clip, ConditionalVAE3, 8) saved
epoch 9
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.05it/s]
Trial (accuracy 0.2701203781555746) for (txt-image, coco, clip, ConditionalVAE3, 9) saved
epoch 10
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.27005563497097695) for (txt-image, coco, clip, ConditionalVAE3, 10) saved
epoch 11
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.02it/s]
Trial (accuracy 0.2685271791832469) for (txt-image, coco, clip, ConditionalVAE3, 11) saved
epoch 12
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.05it/s]
Trial (accuracy 0.26932201492452174) for (txt-image, coco, clip, ConditionalVAE3, 12) saved
epoch 13
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.03it/s]
Trial (accuracy 0.2705807369624343) for (txt-image, coco, clip, ConditionalVAE3, 13) saved
epoch 14
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.03it/s]
Trial (accuracy 0.26877994772652597) for (txt-image, coco, clip, ConditionalVAE3, 14) saved
epoch 15
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.01it/s]
Trial (accuracy 0.26724592433465977) for (txt-image, coco, clip, ConditionalVAE3, 15) saved
epoch 16
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.02it/s]
Trial (accuracy 0.2664795112966377) for (txt-image, coco, clip, ConditionalVAE3, 16) saved
epoch 17
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:29<00:00, 17.06it/s]
Trial (accuracy 0.26581833855459625) for (txt-image, coco, clip, ConditionalVAE3, 17) saved
epoch 18
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.01it/s]
Trial (accuracy 0.26657105290778327) for (txt-image, coco, clip, ConditionalVAE3, 18) saved
epoch 19
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.2660939376688449) for (txt-image, coco, clip, ConditionalVAE3, 19) saved
epoch 20
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.00it/s]
Trial (accuracy 0.2655524542977877) for (txt-image, coco, clip, ConditionalVAE3, 20) saved
epoch 21
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.2673907696019823) for (txt-image, coco, clip, ConditionalVAE3, 21) saved
epoch 22
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.2651423379773291) for (txt-image, coco, clip, ConditionalVAE3, 22) saved
epoch 23
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.00it/s]
Trial (accuracy 0.2657144244898145) for (txt-image, coco, clip, ConditionalVAE3, 23) saved
epoch 24
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.00it/s]
Trial (accuracy 0.2652282874561916) for (txt-image, coco, clip, ConditionalVAE3, 24) saved
epoch 25
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.99it/s]
Trial (accuracy 0.26656812122380624) for (txt-image, coco, clip, ConditionalVAE3, 25) saved
epoch 26
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.00it/s]
Trial (accuracy 0.26483260252765406) for (txt-image, coco, clip, ConditionalVAE3, 26) saved
epoch 27
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.93it/s]
Trial (accuracy 0.2645639896571079) for (txt-image, coco, clip, ConditionalVAE3, 27) saved
epoch 28
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 17.00it/s]
Trial (accuracy 0.2642094367196627) for (txt-image, coco, clip, ConditionalVAE3, 28) saved
epoch 29
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.97it/s]
Trial (accuracy 0.26533705353068415) for (txt-image, coco, clip, ConditionalVAE3, 29) saved
epoch 30
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.98it/s]
Trial (accuracy 0.2671876870984229) for (txt-image, coco, clip, ConditionalVAE3, 30) saved
epoch 31
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.98it/s]
Trial (accuracy 0.2646689592878395) for (txt-image, coco, clip, ConditionalVAE3, 31) saved
epoch 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.97it/s]
Trial (accuracy 0.2640253007835317) for (txt-image, coco, clip, ConditionalVAE3, 32) saved
epoch 33
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.96it/s]
Trial (accuracy 0.2654954674266209) for (txt-image, coco, clip, ConditionalVAE3, 33) saved
epoch 34
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.98it/s]
Trial (accuracy 0.2660708024925161) for (txt-image, coco, clip, ConditionalVAE3, 34) saved
epoch 35
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.97it/s]
Trial (accuracy 0.2670195415175964) for (txt-image, coco, clip, ConditionalVAE3, 35) saved
epoch 36
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.94it/s]
Trial (accuracy 0.2664005241037529) for (txt-image, coco, clip, ConditionalVAE3, 36) saved
epoch 37
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.97it/s]
Trial (accuracy 0.2651982564480505) for (txt-image, coco, clip, ConditionalVAE3, 37) saved
epoch 38
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.95it/s]
Trial (accuracy 0.26558898811696846) for (txt-image, coco, clip, ConditionalVAE3, 38) saved
epoch 39
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.98it/s]
Trial (accuracy 0.2650994496568341) for (txt-image, coco, clip, ConditionalVAE3, 39) saved
epoch 40
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1532/1532 [01:30<00:00, 16.97it/s]
Trial (accuracy 0.26543176611784464) for (txt-image, coco, clip, ConditionalVAE3, 40) saved
"""


# --- Data Extraction and Preparation ---
def parse_logs(log_text):
    records = {}
    current_epoch = None
    for line in log_text.strip().split('\n'):
        epoch_match = re.search(r"^epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in records:
                records[current_epoch] = {}
            continue

        duration_match = re.search(r"\[(\d{2}):(\d{2})<", line)
        if duration_match and current_epoch:
            minutes = int(duration_match.group(1))
            seconds = int(duration_match.group(2))
            records[current_epoch]['duration'] = minutes * 60 + seconds

        accuracy_match = re.search(r"Trial \(accuracy ([\d\.]+)\)", line)
        if accuracy_match and current_epoch:
            records[current_epoch]['accuracy'] = float(accuracy_match.group(1))

    data_list = [{'epoch': k, **v} for k, v in records.items() if 'accuracy' in v and 'duration' in v]
    df = pd.DataFrame(data_list)
    df.sort_values(by='epoch', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df = parse_logs(log_data)

# --- Calculate Derived Metrics ---
df['best_score_so_far'] = df['accuracy'].cummax()
df['score_improvement'] = df['accuracy'].diff().fillna(0)

# --- Generate Chart 1: Performance Over Time ---
print("Generating Chart 1: Performance vs. Epoch...")
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(df['epoch'], df['accuracy'], marker='o', linestyle='-', label='CLIP Score per Epoch', color='royalblue',
         markersize=5)
ax1.plot(df['epoch'], df['best_score_so_far'], linestyle='--', label='Best Score So Far', color='crimson')
best_epoch_data = df.loc[df['accuracy'].idxmax()]
ax1.plot(best_epoch_data['epoch'], best_epoch_data['accuracy'], marker='*', color='gold', markersize=15,
         label=f"Best Score: {best_epoch_data['accuracy']:.4f} at Epoch {int(best_epoch_data['epoch'])}")
ax1.set_title('ConditionalVAE3: Performance vs. Epoch (LR = 1e-4)', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('CLIP Score', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_xticks(range(0, 41, 5))
ax1.set_xlim(0, 41)
plt.tight_layout()
plt.savefig('chart_performance_vs_epoch.png')
print("Saved: chart_performance_vs_epoch.png")

# --- Generate Chart 2: Improvement per Epoch ---
print("Generating Chart 2: Epoch-to-Epoch Score Improvement...")
fig2, ax2 = plt.subplots(figsize=(12, 7))
colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df['score_improvement']]
ax2.bar(df['epoch'], df['score_improvement'], color=colors, width=0.8)
ax2.axhline(0, color='grey', linewidth=0.8)
ax2.set_title('ConditionalVAE3: Epoch-to-Epoch Score Improvement', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Change in CLIP Score', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xticks(range(0, 41, 5))
ax2.set_xlim(0, 41)
plt.tight_layout()
plt.savefig('chart_improvement_per_epoch.png')
print("Saved: chart_improvement_per_epoch.png")

# --- Generate Chart 3: Time per Epoch ---
print("Generating Chart 3: Training Time Consistency...")
fig3, ax3 = plt.subplots(figsize=(12, 7))
ax3.bar(df['epoch'], df['duration'], color='skyblue', width=0.8, label='Time per Epoch')
average_time = df['duration'].mean()
ax3.axhline(average_time, color='r', linestyle='--', linewidth=2, label=f'Average Time: {average_time:.2f}s')
ax3.set_title('ConditionalVAE3: Training Time Consistency', fontsize=16, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Time (seconds)', fontsize=12)
ax3.legend()
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.set_xticks(range(0, 41, 5))
ax3.set_xlim(0, 41)
plt.tight_layout()
plt.savefig('chart_time_per_epoch.png')
print("Saved: chart_time_per_epoch.png")

# Display the charts
plt.show()