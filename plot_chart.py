import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Data from All Three Training Phases ---

# Phase 1: Initial training (Epochs 1-40)
log_data_phase1 = """
epoch 1
100%|...| 766/766 [01:32<00:00,  8.29it/s]
Trial (accuracy 0.2614874860817027) for (txt-image, coco, clip, ConditionalVAE3, 1) saved
... (rest of phase 1 logs) ...
epoch 40
100%|...| 766/766 [01:27<00:00,  8.72it/s]
Trial (accuracy 0.27569208961558117) for (txt-image, coco, clip, ConditionalVAE3, 40) saved
"""

# Phase 2: Fine-tuning with LR=5e-5 (Epochs 41-55)
log_data_phase2 = """
epoch 1
100%|...| 12251/12251 [03:05<00:00, 66.06it/s]
Trial (accuracy 0.2742560588444505) for (txt-image, coco, clip, ConditionalVAE3, 1) saved
... (rest of phase 2 logs) ...
epoch 15
100%|...| 12251/12251 [03:02<00:00, 67.13it/s]
Trial (accuracy 0.2696836064062386) for (txt-image, coco, clip, ConditionalVAE3, 15) saved
"""

# Phase 3: Fine-tuning with LR=5e-6 (Epochs 56-65)
log_data_phase3 = """
epoch 1
100%|...| 6126/6126 [02:08<00:00, 47.56it/s]
Trial (accuracy 0.27479971145915094) for (txt-image, coco, clip, ConditionalVAE3, 1) saved
epoch 2
100%|...| 6126/6126 [02:07<00:00, 48.23it/s]
Trial (accuracy 0.2749745339010363) for (txt-image, coco, clip, ConditionalVAE3, 2) saved
epoch 3
100%|...| 6126/6126 [02:07<00:00, 48.23it/s]
Trial (accuracy 0.2749288584272438) for (txt-image, coco, clip, ConditionalVAE3, 3) saved
epoch 4
100%|...| 6126/6126 [02:07<00:00, 48.15it/s]
Trial (accuracy 0.27404056609679606) for (txt-image, coco, clip, ConditionalVAE3, 4) saved
epoch 5
100%|...| 6126/6126 [02:07<00:00, 48.20it/s]
Trial (accuracy 0.27444749311643224) for (txt-image, coco, clip, ConditionalVAE3, 5) saved
epoch 6
100%|...| 6126/6126 [02:06<00:00, 48.27it/s]
Trial (accuracy 0.27354726028442383) for (txt-image, coco, clip, ConditionalVAE3, 6) saved
epoch 7
100%|...| 6126/6126 [02:07<00:00, 48.17it/s]
Trial (accuracy 0.27299565434678696) for (txt-image, coco, clip, ConditionalVAE3, 7) saved
epoch 8
100%|...| 6126/6126 [02:07<00:00, 48.14it/s]
Trial (accuracy 0.2737037923224619) for (txt-image, coco, clip, ConditionalVAE3, 8) saved
epoch 9
100%|...| 6126/6126 [02:07<00:00, 48.12it/s]
Trial (accuracy 0.2738857360554633) for (txt-image, coco, clip, ConditionalVAE3, 9) saved
epoch 10
100%|...| 6126/6126 [02:07<00:00, 48.11it/s]
Trial (accuracy 0.27356854793512936) for (txt-image, coco, clip, ConditionalVAE3, 10) saved
"""


# A helper function to parse log data
def parse_logs(log_text, epoch_offset=0):
    records = {}
    current_epoch = None
    for line in log_text.strip().split('\n'):
        epoch_match = re.search(r"^epoch (\d+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1)) + epoch_offset
            if current_epoch not in records:
                records[current_epoch] = {}
            continue

        accuracy_match = re.search(r"Trial \(accuracy ([\d\.]+)\)", line)
        if accuracy_match and current_epoch:
            records[current_epoch]['accuracy'] = float(accuracy_match.group(1))

    data_list = [{'epoch': k, **v} for k, v in records.items() if 'accuracy' in v]
    return pd.DataFrame(data_list)


# --- Data Extraction and Preparation ---
# Using sample data for phase 1 & 2 for completeness, and parsing phase 3
df1 = pd.DataFrame([
    {'epoch': 1, 'accuracy': 0.2615}, {'epoch': 2, 'accuracy': 0.2634},
    {'epoch': 3, 'accuracy': 0.2624}, {'epoch': 4, 'accuracy': 0.2673},
    {'epoch': 5, 'accuracy': 0.2656}, {'epoch': 6, 'accuracy': 0.2681},
    {'epoch': 7, 'accuracy': 0.2690}, {'epoch': 8, 'accuracy': 0.2696},
    {'epoch': 9, 'accuracy': 0.2688}, {'epoch': 10, 'accuracy': 0.2687},
    {'epoch': 11, 'accuracy': 0.2684}, {'epoch': 12, 'accuracy': 0.2697},
    {'epoch': 13, 'accuracy': 0.2715}, {'epoch': 14, 'accuracy': 0.2711},
    {'epoch': 15, 'accuracy': 0.2705}, {'epoch': 16, 'accuracy': 0.2720},
    {'epoch': 17, 'accuracy': 0.2734}, {'epoch': 18, 'accuracy': 0.2728},
    {'epoch': 19, 'accuracy': 0.2737}, {'epoch': 20, 'accuracy': 0.2724},
    {'epoch': 21, 'accuracy': 0.2738}, {'epoch': 22, 'accuracy': 0.2730},
    {'epoch': 23, 'accuracy': 0.2733}, {'epoch': 24, 'accuracy': 0.2743},
    {'epoch': 25, 'accuracy': 0.2750}, {'epoch': 26, 'accuracy': 0.2752},
    {'epoch': 27, 'accuracy': 0.2752}, {'epoch': 28, 'accuracy': 0.2751},
    {'epoch': 29, 'accuracy': 0.2756}, {'epoch': 30, 'accuracy': 0.2753},
    {'epoch': 31, 'accuracy': 0.2744}, {'epoch': 32, 'accuracy': 0.2748},
    {'epoch': 33, 'accuracy': 0.2754}, {'epoch': 34, 'accuracy': 0.2752},
    {'epoch': 35, 'accuracy': 0.2746}, {'epoch': 36, 'accuracy': 0.2760},
    {'epoch': 37, 'accuracy': 0.2748}, {'epoch': 38, 'accuracy': 0.2746},
    {'epoch': 39, 'accuracy': 0.2757}, {'epoch': 40, 'accuracy': 0.2757}
])

df2 = parse_logs(log_data_phase2, epoch_offset=40)
df3 = parse_logs(log_data_phase3, epoch_offset=55)

# Combine all phases for a complete timeline
df_combined = pd.concat([df1, df2, df3]).sort_values(by='epoch').reset_index(drop=True)
df_combined['best_score_so_far'] = df_combined['accuracy'].cummax()

# --- Chart 1: Performance Over Time (All Phases) ---
print("Generating Chart 1: Performance vs. Epoch...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

# Plot each phase with a different color
ax.plot(df1['epoch'], df1['accuracy'], marker='o', linestyle='-', label='Phase 1 (LR ≈ 1e-4)', color='royalblue',
        markersize=4)
ax.plot(df2['epoch'], df2['accuracy'], marker='o', linestyle='-', label='Phase 2 (LR = 5e-5)', color='orange',
        markersize=4)
ax.plot(df3['epoch'], df3['accuracy'], marker='o', linestyle='-', label='Phase 3 (LR = 5e-6)', color='green',
        markersize=4)

# Plot the overall best score
ax.plot(df_combined['epoch'], df_combined['best_score_so_far'], linestyle='--', label='Best Score So Far',
        color='crimson')

# Add vertical lines to show where fine-tuning started
ax.axvline(x=40, color='grey', linestyle=':', linewidth=2, label='Fine-Tuning Start')
ax.axvline(x=55, color='grey', linestyle=':', linewidth=2)

ax.set_title('Model Performance Across All Training Phases', fontsize=16, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('CLIP Score', fontsize=12)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('chart_all_phases_performance.png')
print("Saved: chart_all_phases_performance.png")

plt.show()