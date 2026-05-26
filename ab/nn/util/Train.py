import importlib
import math
import os
import platform
import psutil
import sys
import time as time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Union
from uuid import uuid4

import torch
from torch.cuda import OutOfMemoryError

import ab.nn.util.CodeEval as codeEvaluator
import ab.nn.util.db.Write as DB_Write
from ab.nn.util.Classes import DataRoll
from ab.nn.util.Exception import *
from ab.nn.util.Loader import load_dataset
from ab.nn.util.Util import *
from ab.nn.util.db.Calc import save_results
from ab.nn.util.db.Read import supported_transformers
from ab.nn.util.db.Util import *
from ab.nn.util.Const import ckpt_dir

debug = False


@dataclass
class EpochMetrics:
    """Stores metrics for a single epoch"""
    epoch: int
    # Loss metrics
    train_loss: float = 0.0
    test_loss: float = 0.0
    # Accuracy metrics
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    # Raw MAE metrics (optional, for regression metrics that expose get_mae)
    train_mae: Optional[float] = None
    test_mae: Optional[float] = None
    # Training dynamics
    lr: float = 0.0
    gradient_norm: float = 0.0
    # Timing
    samples_per_second: float = 0.0


def compute_gradient_norm(model) -> float:
    """Compute the L2 norm of gradients"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def get_current_lr(optimizer) -> Optional[float]:
    """Get current learning rate from optimizer"""
    if optimizer:
        for param_group in optimizer.param_groups:
            return param_group['lr']
    return None


def get_gpu_memory_kb() -> Optional[float]:
    """Get current GPU memory usage in KB"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024
    return None


def get_system_info() -> dict:
    """Collect comprehensive system information"""
    info = {
        'cpu_type': platform.processor() or platform.machine(),
        'cpu_count': psutil.cpu_count(logical=True),
        'total_ram_kb': round(psutil.virtual_memory().total / 1024, 2),
    }

    if torch.cuda.is_available():
        try:
            gpu_props = torch.cuda.get_device_properties(0)
            info['gpu_type'] = gpu_props.name
            info['gpu_total_memory_kb'] = round(gpu_props.total_memory / 1024, 2)
        except Exception:
            info['gpu_type'] = 'CUDA Available (details unavailable)'
    else:
        info['gpu_type'] = 'No GPU'

    return info


def get_current_resource_usage() -> dict:
    """Get current resource usage metrics"""
    usage = {
        'occupied_ram_kb': round(psutil.virtual_memory().used / 1024, 2),
        'ram_usage_percent': psutil.virtual_memory().percent,
        'cpu_usage_percent': psutil.cpu_percent(interval=0.1),
    }

    if torch.cuda.is_available():
        try:
            occupied_gpu_kb = torch.cuda.memory_allocated() / 1024
            usage['occupied_gpu_memory_kb'] = round(occupied_gpu_kb, 2)
            usage['gpu_memory_usage_percent'] = round(
                (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100, 2
            )
        except Exception:
            pass

    return usage


def _ensure_dir(p: Union[str, Path]) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p


def _make_tb_run_dir(config: tuple[str, str, str, str], trial_number: Optional[int] = None) -> str:
    """
    Create a unique TensorBoard run directory so trials don't overwrite each other.
    runs/<dataset>/<model>/trial_<N>_<timestamp>
    """
    task, dataset_name, metric, nn = config
    ts = time.strftime("%Y%m%d-%H%M%S")
    if trial_number is None:
        run_name = f"{dataset_name}/{nn}/{ts}"
    else:
        run_name = f"{dataset_name}/{nn}/trial_{trial_number}_{ts}"
    return _ensure_dir(os.path.join("runs", run_name))


def optuna_objective(trial, config, nn_prm, num_workers, min_lr, max_lr, min_momentum, max_momentum, min_dropout,
                     max_dropout, min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations,
                     epoch_max, pretrained, epoch_limit_minutes, save_pth_weights, save_onnx_weights):
    task, dataset_name, metric, nn = config
    try:
        s_prm: set = get_ab_nn_attr(f"nn.{nn}", "supported_hyperparameters")()

        prms = dict(nn_prm)
        for prm in s_prm:
            if not (prm in prms and prms[prm] is not None):
                match prm:
                    case 'lr':
                        prms[prm] = trial.suggest_float(prm, min_lr, max_lr, log=True)
                    case 'momentum':
                        prms[prm] = trial.suggest_float(prm, min_momentum, max_momentum)
                    case 'dropout':
                        prms[prm] = trial.suggest_float(prm, min_dropout, max_dropout)
                    case 'pretrained':
                        prms[prm] = float(pretrained if pretrained else trial.suggest_categorical(prm, [0, 1]))
                    case _:
                        prms[prm] = trial.suggest_float(prm, 0.0, 1.0)

        prms['epoch_max'] = epoch_max

        batch = add_categorical_if_absent(
            trial,
            prms,
            'batch',
            lambda: [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)]
        )
        transform_name = add_categorical_if_absent(
            trial,
            prms,
            'transform',
            supported_transformers,
            default=transform
        )

        prm_str = ''
        for k, v in prms.items():
            prm_str += f", {k}: {v}"
        print(f"Initialize training with {prm_str[2:]}")

        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset_name, transform_name)

        tb_log_dir = _make_tb_run_dir(config, trial_number=trial.number)

        trainer = Train(
            config=config,
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=batch,
            nn_module=nn_mod('nn', nn),
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=num_workers,
            prm=prms,
            tb_log_dir=tb_log_dir
        )

        return trainer.train_n_eval(
            epoch_max,
            epoch_limit_minutes,
            save_pth_weights,
            save_onnx_weights,
            train_set
        )

    except Exception as e:
        accuracy_duration = 0.0, 0.0, 1
        if isinstance(e, OutOfMemoryError):
            if max_batch_binary_power_local <= min_batch_binary_power:
                return accuracy_duration
            else:
                raise CudaOutOfMemory(batch)
        elif isinstance(e, AccuracyException):
            print(e.message)
            return e.accuracy, accuracy_to_time_metric(e.accuracy, minimum_accuracy, e.duration), e.duration
        elif isinstance(e, LearnTimeException):
            print(
                f"Estimated training time, minutes: {format_time(e.estimated_training_time)}, "
                f"but limit {format_time(e.epoch_limit_minutes)}."
            )
            return (e.epoch_limit_minutes / e.estimated_training_time) / 1e5, 0, e.duration
        else:
            print(f"error '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return accuracy_duration
            else:
                raise e


class Train:
    def __init__(self, config: tuple[str, str, str, str], out_shape: tuple, minimum_accuracy: float,
                 batch: int, nn_module, task, train_dataset, test_dataset, metric, num_workers,
                 prm: dict, save_to_db=True, is_code=False, tb_log_dir: str = "runs/experiment_1"):
        """
        Universal class for training CV, text generation and other models.
        Preserves multi-metric framework behaviour while adding better regression tracking.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.minimum_accuracy = minimum_accuracy

        self.out_shape = out_shape
        self.batch = batch
        self.task = task
        self.prm = prm

        self.metric_name = metric
        self.metric_names = [m.strip() for m in metric.split(',')]
        self.metric_fns = [self.load_metric_function(m) for m in self.metric_names]
        self.primary_metric_fn = self.metric_fns[0]

        self.save_to_db = save_to_db
        self.is_code = is_code

        self.num_workers = num_workers
        self.train_loader = train_loader_f(self.train_dataset, self.batch, num_workers)
        self.test_loader = test_loader_f(self.test_dataset, self.batch, num_workers)

        self.in_shape = get_in_shape(train_dataset, num_workers)
        self.device = torch_device()

        model_net = get_attr(nn_module, 'Net')
        self.model_name = nn_module
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)

        self.loss_fn = self._get_loss_function()

        self.epoch_history: List[EpochMetrics] = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.best_state_dict = None
        self.save_path = None

        self.system_info = get_system_info()
        
        from ab.nn.util.Const import stat_train_dir
        self.train_stat_root = _ensure_dir(os.path.join(str(stat_train_dir), '_'.join(self.config)))

        # Keep train artifacts together under the per-run stat folder.
        self.viz_dir = _ensure_dir(os.path.join(self.train_stat_root, 'plots'))
        self.tb_log_dir = _ensure_dir(os.path.join(self.train_stat_root, 'tensorboard', tb_log_dir.replace('runs/', '')))
        self.metrics_dir = _ensure_dir(self.train_stat_root)
        self.models_dir = _ensure_dir(os.path.join(self.train_stat_root, 'models'))

    def _get_loss_function(self):
        """Get loss function for metric tracking."""
        if hasattr(self.model, 'criterion'):
            return self.model.criterion
        elif hasattr(self.model, 'loss_fn'):
            return self.model.loss_fn
        elif hasattr(self.model, 'criteria') and len(self.model.criteria) > 0:
            return self.model.criteria[0]
        else:
            if 'classification' in self.task or 'img-class' in self.task:
                return torch.nn.CrossEntropyLoss()
            elif 'segmentation' in self.task:
                return torch.nn.CrossEntropyLoss()
            else:
                return torch.nn.MSELoss()

    def _compute_loss(self, data_loader) -> float:
        """Compute average loss over a dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                try:
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, labels)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception:
                    pass

        return total_loss / max(num_batches, 1)

    def _metric_supports_mae(self, metric_fn) -> bool:
        return hasattr(metric_fn, 'get_mae') and callable(getattr(metric_fn, 'get_mae'))

    def _compute_accuracy(self, data_loader) -> float:
        """Compute primary metric over a dataset."""
        self.model.eval()
        self.primary_metric_fn.reset()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.primary_metric_fn(outputs, labels)

        return self.primary_metric_fn.result()

    def _compute_primary_metric_and_mae(self, data_loader):
        """
        Returns:
            primary_metric_value, raw_mae_or_none
        """
        self.model.eval()
        self.primary_metric_fn.reset()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.primary_metric_fn(outputs, labels)

        primary_value = self.primary_metric_fn.result()
        raw_mae = self.primary_metric_fn.get_mae() if self._metric_supports_mae(self.primary_metric_fn) else None
        return primary_value, raw_mae

    def load_metric_function(self, metric_name):
        try:
            module = importlib.import_module(nn_mod('metric', metric_name))
            return module.create_metric(self.out_shape)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(
                f"Metric '{metric_name}' not found. Ensure a corresponding file and function exist. "
                f"Ensure the metric module has create_metric()"
            ) from e

    def train_n_eval(self, epoch_max, epoch_limit_minutes, save_pth_weights, save_onnx_weights, train_set,
                     save_path: Union[str, Path] = None):
        """Training and evaluation with comprehensive metrics tracking."""

        if save_path is None and not self.is_code:
            # Save model stats to the per-run stat folder.
            save_path = self.metrics_dir
        self.save_path = save_path

        start_time = time.time_ns()
        self.model.train_setup(self.prm)
        accuracy_to_time = 0.0
        duration = sys.maxsize
        accuracy = 0.0

        tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=self.tb_log_dir)
        except ImportError:
            tb_writer = None

        for epoch in range(1, epoch_max + 1):
            epoch_start_time = time.time_ns()
            print(f"epoch {epoch}", flush=True)

            self.model.train()
            self.model.learn(DataRoll(self.train_loader, epoch_limit_minutes))

            grad_norm = compute_gradient_norm(self.model)

            current_optimizer = getattr(self.model, 'optimizer', None)
            lr_now = get_current_lr(current_optimizer)

            train_loss = self._compute_loss(self.train_loader)
            test_loss = self._compute_loss(self.test_loader)

            train_accuracy, train_mae = self._compute_primary_metric_and_mae(self.train_loader)
            test_accuracy, test_mae = self._compute_primary_metric_and_mae(self.test_loader)

            accuracy = test_accuracy
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            epoch_duration = (time.time_ns() - epoch_start_time) / 1e9

            total_samples = len(self.train_dataset)
            samples_per_second = total_samples / max(epoch_duration, 0.001)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_accuracy=train_accuracy,
                test_accuracy=accuracy,
                train_mae=train_mae,
                test_mae=test_mae,
                lr=lr_now if lr_now is not None else 0.0,
                gradient_norm=grad_norm,
                samples_per_second=samples_per_second
            )
            self.epoch_history.append(epoch_metrics)

            print(f"  Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"  Train Acc: {train_accuracy:.4f}, Test Acc: {accuracy:.4f}")
            if train_mae is not None or test_mae is not None:
                train_mae_txt = f"{train_mae:.3f} yrs" if train_mae is not None else "n/a"
                test_mae_txt = f"{test_mae:.3f} yrs" if test_mae is not None else "n/a"
                print(f"  Train MAE: {train_mae_txt}, Test MAE: {test_mae_txt}")
            if lr_now is not None:
                print(f"  LR: {lr_now:.6f}, Grad Norm: {grad_norm:.4f}, Throughput: {samples_per_second:.1f} samples/s")

            if tb_writer:
                tb_writer.add_scalar('Loss/train', train_loss, epoch)
                tb_writer.add_scalar('Loss/val', test_loss, epoch)
                tb_writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                tb_writer.add_scalar('Accuracy/val', accuracy, epoch)
                if train_mae is not None:
                    tb_writer.add_scalar('MAE/train_years', train_mae, epoch)
                if test_mae is not None:
                    tb_writer.add_scalar('MAE/val_years', test_mae, epoch)
                if lr_now is not None:
                    tb_writer.add_scalar('Learning_Rate', lr_now, epoch)
                tb_writer.add_scalar('Gradient_Norm', grad_norm, epoch)
                tb_writer.add_scalar('Throughput', samples_per_second, epoch)

                try:
                    pred_list = []
                    true_list = []
                    img_list = []
                    self.model.eval()
                    with torch.no_grad():
                        for batch in self.test_loader:
                            inputs, labels = batch
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            outputs = self.model(inputs)

                            if outputs.ndim > 1 and outputs.shape[1] > 1:
                                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                            else:
                                preds = outputs.squeeze().detach().cpu().numpy()

                            true = labels.squeeze().detach().cpu().numpy()
                            pred_list.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)
                            true_list.extend(true.tolist() if hasattr(true, 'tolist') else true)

                            take = min(8 - len(img_list), inputs.shape[0])
                            if take > 0:
                                img_list.extend(inputs[:take].detach().cpu())

                            if len(img_list) >= 8 and len(pred_list) >= 64:
                                break

                    import numpy as np
                    import matplotlib.pyplot as plt

                    pred_arr = np.array(pred_list)
                    true_arr = np.array(true_list)

                    if len(pred_arr) > 0 and len(true_arr) > 0:
                        fig = plt.figure(figsize=(5, 5))
                        plt.scatter(true_arr, pred_arr, alpha=0.5)
                        plt.xlabel('True')
                        plt.ylabel('Predicted')
                        plt.title('Predicted vs True')
                        tb_writer.add_figure('Scatter/Pred_vs_True', fig, epoch)
                        # also save PNG
                        try:
                            scatter_path = os.path.join(self.viz_dir, f'scatter_epoch_{epoch}.png')
                            fig.savefig(scatter_path)
                        except Exception:
                            pass
                        plt.close(fig)

                    if len(pred_arr) > 0:
                        fig_hist = plt.figure(figsize=(6, 3))
                        plt.hist(pred_arr, bins=20, alpha=0.7, label='Predicted')
                        plt.hist(true_arr, bins=20, alpha=0.5, label='True')
                        plt.legend()
                        plt.title('Prediction Distribution')
                        tb_writer.add_figure('Histogram/Pred_Distribution', fig_hist, epoch)
                        try:
                            hist_path = os.path.join(self.viz_dir, f'hist_epoch_{epoch}.png')
                            fig_hist.savefig(hist_path)
                        except Exception:
                            pass
                        plt.close(fig_hist)

                    if len(img_list) > 0:
                        import torchvision
                        grid = torchvision.utils.make_grid(img_list, nrow=4, normalize=True)
                        tb_writer.add_image('Samples/Images', grid, epoch)

                        labels_text = []
                        for p, t in zip(pred_arr[:len(img_list)], true_arr[:len(img_list)]):
                            try:
                                labels_text.append(f"P:{int(round(float(p)))} T:{int(round(float(t)))}")
                            except Exception:
                                labels_text.append(f"P:{p} T:{t}")

                        n_show = min(len(img_list), len(labels_text))
                        ncols = 4
                        nrows = max(1, math.ceil(n_show / ncols))
                        fig_img, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
                        axes = np.array(axes, dtype=object).reshape(nrows, ncols)

                        for idx, ax in enumerate(axes.flat):
                            ax.set_xticks([])
                            ax.set_yticks([])
                            if idx >= n_show:
                                ax.axis('off')
                                continue

                            img_np = img_list[idx].detach().cpu().permute(1, 2, 0).numpy()
                            img_min = float(np.min(img_np))
                            img_max = float(np.max(img_np))
                            if img_max > img_min:
                                img_np = (img_np - img_min) / (img_max - img_min)
                            else:
                                img_np = np.zeros_like(img_np)

                            ax.imshow(img_np)
                            ax.text(
                                0.02,
                                0.98,
                                labels_text[idx],
                                transform=ax.transAxes,
                                va='top',
                                ha='left',
                                fontsize=8,
                                color='white',
                                bbox={
                                    'facecolor': 'black',
                                    'alpha': 0.65,
                                    'pad': 2,
                                    'edgecolor': 'none'
                                }
                            )

                        fig_img.tight_layout(pad=0.6)
                        tb_writer.add_figure('Samples/Images_with_Labels', fig_img, epoch)
                        try:
                            img_path = os.path.join(self.viz_dir, f'samples_epoch_{epoch}.png')
                            # save the grid as an image
                            torchvision.utils.save_image(grid, os.path.join(self.viz_dir, f'grid_epoch_{epoch}.png'), normalize=True)
                            fig_img.savefig(os.path.join(self.viz_dir, f'samples_with_labels_epoch_{epoch}.png'))
                        except Exception:
                            pass
                        plt.close(fig_img)

                except Exception as viz_e:
                    print(f"[WARN] TensorBoard viz failed (epoch {epoch}): {viz_e}")

                tb_writer.flush()

            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if not good(accuracy, self.minimum_accuracy, duration):
                raise AccuracyException(
                    accuracy, duration,
                    f"Accuracy is too low: {accuracy}."
                    f" The minimum accepted accuracy for the '{self.config[1]}' dataset is {self.minimum_accuracy}."
                )

            if save_pth_weights or save_onnx_weights:
                save_if_best(
                    self.model, self.model_name, accuracy, save_pth_weights, save_onnx_weights,
                    train_set, self.num_workers, save_path=self.models_dir
                )

            only_prm = {k: v for k, v in self.prm.items() if k not in {'uid', 'duration', 'accuracy', 'epoch'}}
            resource_usage = get_current_resource_usage()

            prm = merge_prm(
                self.prm,
                {
                    'uid': uuid4(only_prm),
                    'duration': duration,
                    'accuracy': accuracy,
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': accuracy,
                    'gradient_norm': grad_norm,
                    'samples_per_second': samples_per_second,
                    'best_accuracy': self.best_accuracy,
                    'best_epoch': self.best_epoch,
                }
                | ({'train_mae': train_mae} if train_mae is not None else {})
                | ({'test_mae': test_mae} if test_mae is not None else {})
                | self.system_info
                | resource_usage
                | ({'lr_now': lr_now} if lr_now is not None else {})
                | ({'gpu_memory_kb': get_gpu_memory_kb()} if get_gpu_memory_kb else {})
            )

            if self.save_to_db:
                if self.is_code:
                    if self.save_path:
                        save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                    else:
                        print(f"[WARN] parameter `save_path` set to null, the statistics will not be saved into a file.")
                else:
                    save_results(self.config + (epoch,), join(self.save_path, f"{epoch}.json"), prm)
                    DB_Write.save_results(self.config + (epoch,), prm)

        if tb_writer:
            tb_writer.close()

        if save_path and self.epoch_history:
            self._save_training_summary()

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
            self.model.eval()

        if hasattr(self.test_dataset, 'held_out_test'):
            test_loader = test_loader_f(self.test_dataset.held_out_test, self.batch, self.num_workers)
            held_out_loss = self._compute_loss(test_loader)
            held_out_primary, held_out_all = self.eval(test_loader)

            print(f"\n{'='*60}")
            print(f"  HELD-OUT TEST SET (20%) — best checkpoint, never used during training")
            print(f"  Test Loss: {held_out_loss:.4f} | Test {self.metric_names[0]}: {held_out_primary:.4f}")

            if self._metric_supports_mae(self.primary_metric_fn):
                held_out_mae = self._compute_primary_metric_and_mae(test_loader)[1]
                if held_out_mae is not None:
                    print(f"  Held-out MAE: {held_out_mae:.3f} yrs")

            if len(self.metric_names) > 1:
                print("  All metrics:")
                for name, value in held_out_all.items():
                    print(f"    {name}: {value:.4f}")

            print(f"{'='*60}", flush=True)

        total_trial_seconds = (time.time_ns() - start_time) / 1e9
        print(f"\n{'='*60}")
        print(f"  Trial complete | {epoch_max} epochs | Total time: {total_trial_seconds/60:.1f} min ({total_trial_seconds:.0f}s)")
        print(f"  Best Val {self.metric_names[0]}: {self.best_accuracy:.4f} @ epoch {self.best_epoch}")
        if any(e.test_mae is not None for e in self.epoch_history):
            valid_maes = [e.test_mae for e in self.epoch_history if e.test_mae is not None]
            if valid_maes:
                print(f"  Best Val MAE: {min(valid_maes):.3f} yrs")
        print(f"{'='*60}\n", flush=True)

        return accuracy, accuracy_to_time, duration

    def _save_training_summary(self):
        """Save comprehensive training summary"""
        import json

        final_resource_usage = get_current_resource_usage()
        out_dir = Path(self.tb_log_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'config': {
                'task': self.config[0],
                'dataset': self.config[1],
                'metric': self.config[2],
                'model': self.config[3] if len(self.config) > 3 else self.model_name,
            },
            'hyperparameters': {k: v for k, v in self.prm.items() if k not in {'uid', 'duration', 'accuracy'}},
            'model_info': {
                'input_shape': list(self.in_shape),
                'output_shape': list(self.out_shape) if hasattr(self.out_shape, '__iter__') else self.out_shape,
            },
            'system_info': self.system_info,
            'training_summary': {
                'total_epochs': len(self.epoch_history),
                'best_accuracy': self.best_accuracy,
                'best_epoch': self.best_epoch,
                'best_val_mae_years': min([e.test_mae for e in self.epoch_history if e.test_mae is not None], default=None),
                'final_train_loss': self.epoch_history[-1].train_loss if self.epoch_history else 0,
                'final_test_loss': self.epoch_history[-1].test_loss if self.epoch_history else 0,
                'final_accuracy': self.epoch_history[-1].test_accuracy if self.epoch_history else 0,
                'final_val_mae_years': self.epoch_history[-1].test_mae if self.epoch_history else None,
                'gpu_memory_kb': get_gpu_memory_kb(),
            },
            'final_resource_usage': final_resource_usage,
            'learning_curves': {
                'epochs': [e.epoch for e in self.epoch_history],
                'train_loss': [e.train_loss for e in self.epoch_history],
                'test_loss': [e.test_loss for e in self.epoch_history],
                'train_accuracy': [e.train_accuracy for e in self.epoch_history],
                'test_accuracy': [e.test_accuracy for e in self.epoch_history],
                'train_mae': [e.train_mae for e in self.epoch_history],
                'test_mae': [e.test_mae for e in self.epoch_history],
                'lr': [e.lr for e in self.epoch_history],
                'gradient_norm': [e.gradient_norm for e in self.epoch_history],
                'samples_per_second': [e.samples_per_second for e in self.epoch_history],
            },
            'epoch_details': [asdict(e) for e in self.epoch_history]
        }

        # Add model parameter count and approximate model size
        try:
            param_count = sum(p.numel() for p in self.model.parameters())
            param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            model_size_mb = param_bytes / 1024.0 / 1024.0
            summary['model_info']['param_count'] = int(param_count)
            summary['model_info']['model_size_mb'] = float(model_size_mb)
            summary['visualization_dir'] = str(self.viz_dir)
        except Exception:
            pass

        summary_path = Path(self.train_stat_root) / 'training_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to {summary_path}")
        except Exception as e:
            print(f"[WARN] Failed to save training summary: {e}")

    def eval(self, test_loader):
        """
        Evaluate all configured metrics.
        Returns:
            (primary_metric_value, all_results_dict)
        """
        if debug:
            for inputs, labels in test_loader:
                print(f"[EVAL DEBUG] labels type: {type(labels)}")
                if isinstance(labels, torch.Tensor):
                    print(f"[EVAL DEBUG] labels shape: {labels.shape}")
                else:
                    print(f"[EVAL DEBUG] labels sample: {labels[:2]}")
                break

        self.model.eval()
        for metric_fn in self.metric_fns:
            metric_fn.reset()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                for metric_fn in self.metric_fns:
                    metric_fn(outputs, labels)

        results = {}
        for name, metric_fn in zip(self.metric_names, self.metric_fns):
            results[name] = metric_fn.result()

        return results[self.metric_names[0]], results


def train_new(nn_code, task, dataset, metric, prm, save_to_db=True, prefix: Union[str, None] = None,
              save_path: Union[str, None] = None, export_onnx=False,
              epoch_limit_minutes=default_epoch_limit_minutes, transform_dir=None):
    """
    Train the model with the given code and hyperparameters and evaluate it.
    """
    model_name = uuid4(nn_code)
    if prefix:
        model_name = prefix + "-" + model_name

    tmp_modul = ".".join((out, 'nn', 'tmp'))
    tmp_modul_name = ".".join((tmp_modul, model_name))
    tmp_dir = ab_root_path / tmp_modul.replace('.', '/')
    create_file(tmp_dir, '__init__.py')
    temp_file_path = tmp_dir / f"{model_name}.py"
    trainer = None

    try:
        with open(temp_file_path, 'w') as f:
            f.write(nn_code)

        res = codeEvaluator.evaluate_single_file(temp_file_path)

        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset, prm['transform'], transform_dir)
        num_workers = prm.get('num_workers', 1)

        tb_log_dir = _make_tb_run_dir((task, dataset, metric, model_name), trial_number=None)

        trainer = Train(
            config=(task, dataset, metric, model_name),
            out_shape=out_shape,
            minimum_accuracy=minimum_accuracy,
            batch=prm['batch'],
            nn_module=tmp_modul_name,
            task=task,
            train_dataset=train_set,
            test_dataset=test_set,
            metric=metric,
            num_workers=num_workers,
            prm=prm,
            save_to_db=save_to_db,
            is_code=True,
            tb_log_dir=tb_log_dir
        )

        epoch = prm['epoch']
        accuracy, accuracy_to_time, duration = trainer.train_n_eval(
            epoch, epoch_limit_minutes, False, export_onnx, train_set, save_path=save_path
        )

        if save_to_db:
            if good(accuracy, minimum_accuracy, duration):
                model_name = DB_Write.save_nn(nn_code, task, dataset, metric, epoch, prm, force_name=model_name)
                print(f"Model saved to database with accuracy: {accuracy}")
            else:
                print(f"Model accuracy {accuracy} is below the minimum threshold {minimum_accuracy}. Not saved.")

    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        remove(temp_file_path)

        try:
            del train_set
        except NameError:
            pass

        try:
            del test_set
        except NameError:
            pass

        try:
            if trainer:
                del trainer.model
        except NameError:
            pass

        release_memory()

    return model_name, accuracy, accuracy_to_time, res['score']