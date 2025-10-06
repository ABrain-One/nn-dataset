import pandas as pd
import matplotlib.pyplot as plt
import re

# --- Data from your stabilized VAE-GAN training phase ---
log_data = """
--- Initializing VAE-GAN with G_LR: 3e-06 | D_LR: 2e-09 | Beta1: 0.9885972119167569 ---
Resuming GAN training. Loading checkpoint from: checkpoints/ConditionalVAE4/best_model.pth
Resumed from Epoch 4955. Best score so far: 0.2747
epoch 1
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:10<00:00, 23.46it/s]
Epoch 4954 - G_Loss: 2.3625, D_Loss: 0.6733
--- New best score: 0.2751 at epoch 5088! Saving checkpoint to checkpoints/ConditionalVAE4/best_model.pth ---
Trial (accuracy 0.27512670470620987) for (txt-image, coco, clip, ConditionalVAE4, 1) saved
epoch 2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.45it/s]
Epoch 5088 - G_Loss: 2.3581, D_Loss: 0.6731
Trial (accuracy 0.2741119001156816) for (txt-image, coco, clip, ConditionalVAE4, 2) saved
epoch 3
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.44it/s]
Epoch 5222 - G_Loss: 2.3543, D_Loss: 0.6730
Trial (accuracy 0.27397855783622954) for (txt-image, coco, clip, ConditionalVAE4, 3) saved
epoch 4
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.37it/s]
Epoch 5356 - G_Loss: 2.3505, D_Loss: 0.6728
Trial (accuracy 0.27359059927396684) for (txt-image, coco, clip, ConditionalVAE4, 4) saved
epoch 5
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.38it/s]
Epoch 5490 - G_Loss: 2.3470, D_Loss: 0.6726
Trial (accuracy 0.27294918252820166) for (txt-image, coco, clip, ConditionalVAE4, 5) saved
epoch 6
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.33it/s]
Epoch 5624 - G_Loss: 2.3433, D_Loss: 0.6725
Trial (accuracy 0.27344580106646105) for (txt-image, coco, clip, ConditionalVAE4, 6) saved
epoch 7
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.43it/s]
Epoch 5758 - G_Loss: 2.3396, D_Loss: 0.6723
Trial (accuracy 0.27373339458715135) for (txt-image, coco, clip, ConditionalVAE4, 7) saved
epoch 8
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.37it/s]
Epoch 5892 - G_Loss: 2.3362, D_Loss: 0.6721
Trial (accuracy 0.27276945310218315) for (txt-image, coco, clip, ConditionalVAE4, 8) saved
epoch 9
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:05<00:00, 24.32it/s]
Epoch 6026 - G_Loss: 2.3327, D_Loss: 0.6720
Trial (accuracy 0.27343462321914247) for (txt-image, coco, clip, ConditionalVAE4, 9) saved
epoch 10
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 3063/3063 [02:06<00:00, 24.29it/s]
Epoch 6160 - G_Loss: 2.3292, D_Loss: 0.6718
Trial (accuracy 0.2728330698458948) for (txt-image, coco, clip, ConditionalVAE4, 10) saved
"""

# --- Data Extraction and Preparation ---
records = {}
current_epoch = None
for line in log_data.strip().split('\n'):
    epoch_match = re.search(r"^epoch (\d+)", line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        if current_epoch not in records:
            records[current_epoch] = {}
        continue

    loss_match = re.search(r"G_Loss: ([\d\.]+), D_Loss: ([\d\.]+)", line)
    if loss_match and current_epoch:
        records[current_epoch]['g_loss'] = float(loss_match.group(1))
        records[current_epoch]['d_loss'] = float(loss_match.group(2))

    accuracy_match = re.search(r"Trial \(accuracy ([\d\.]+)\)", line)
    if accuracy_match and current_epoch:
        records[current_epoch]['accuracy'] = float(accuracy_match.group(1))

# Convert dictionary to list of dictionaries for DataFrame
data_list = [{'epoch': k, **v} for k, v in records.items() if all(key in v for key in ['g_loss', 'd_loss', 'accuracy'])]
df = pd.DataFrame(data_list)
df.sort_values(by='epoch', inplace=True)
df.reset_index(drop=True, inplace=True)

# Calculate the best score so far for this run
df['best_score_so_far'] = df['accuracy'].cummax()


# --- Chart 1: Generator vs. Discriminator Loss ---
print("Generating Chart 1: Generator vs. Discriminator Loss...")
plt.style.use('seaborn-v0_8-whitegrid')
fig1, ax1 = plt.subplots(figsize=(12, 7))

ax1.plot(df['epoch'], df['g_loss'], marker='o', linestyle='-', label='Generator Loss (G_Loss)', color='blue')
ax1.plot(df['epoch'], df['d_loss'], marker='o', linestyle='-', label='Discriminator Loss (D_Loss)', color='red')

ax1.set_title('VAE-GAN: Stable Training Loss Dynamics', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend()
ax1.grid(True)
ax1.set_xticks(range(1, 11))
ax1.set_xlim(0.5, 10.5)

plt.tight_layout()
plt.savefig('chart_gan_losses_stable.png')
print("Saved: chart_gan_losses_stable.png")


# --- Chart 2: Model Performance (CLIP Score) During Fine-Tuning ---
print("Generating Chart 2: CLIP Score During Fine-Tuning...")
fig2, ax2 = plt.subplots(figsize=(12, 7))

ax2.plot(df['epoch'], df['accuracy'], marker='o', linestyle='-', label='CLIP Score per Epoch', color='green')
ax2.plot(df['epoch'], df['best_score_so_far'], linestyle='--', label='Best Score in this Run', color='orange')
# Add a line for the previous best score for context
ax2.axhline(0.2747, color='purple', linestyle=':', linewidth=2, label='Previous Best Score (0.2747)')

# Highlight the new best score
best_epoch_data = df.loc[df['accuracy'].idxmax()]
ax2.plot(best_epoch_data['epoch'], best_epoch_data['accuracy'], marker='*', color='gold', markersize=15,
         label=f"New Best Score: {best_epoch_data['accuracy']:.4f}")

ax2.set_title('VAE-GAN: CLIP Score Improvement', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('CLIP Score', fontsize=12)
ax2.legend()
ax2.grid(True)
ax2.set_xticks(range(1, 11))
ax2.set_xlim(0.5, 10.5)

plt.tight_layout()
plt.savefig('chart_gan_clip_score_stable.png')
print("Saved: chart_gan_clip_score_stable.png")

# Display the charts
plt.show()