import importlib
import sys
import time as time
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

import os
import math
from pathlib import Path
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
                     max_dropout, min_batch_binary_power, max_batch_binary_power_local, transform, fail_iterations, epoch_max,
                     pretrained, epoch_limit_minutes, save_pth_weights, save_onnx_weights):
    task, dataset_name, metric, nn = config
    try:
        # Load model
        s_prm: set = get_ab_nn_attr(f"nn.{nn}", "supported_hyperparameters")()
        # Suggest hyperparameters
        prms = dict(nn_prm)
        for prm in s_prm:
            if not (prm in prms and prms[prm]):
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
            trial, prms, 'batch',
            lambda: [max_batch(x) for x in range(min_batch_binary_power, max_batch_binary_power_local + 1)]
        )
        transform_name = add_categorical_if_absent(trial, prms, 'transform', supported_transformers, default=transform)

        prm_str = ''
        for k, v in prms.items():
            prm_str += f", {k}: {v}"
        print(f"Initialize training with {prm_str[2:]}")

        # Load dataset
        out_shape, minimum_accuracy, train_set, test_set = load_dataset(task, dataset_name, transform_name)

        # ✅ Unique TensorBoard logdir per trial
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

        return trainer.train_n_eval(epoch_max, epoch_limit_minutes, save_pth_weights, save_onnx_weights, train_set)

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
            print(f"Estimated training time, minutes: {format_time(e.estimated_training_time)}, but limit {format_time(e.epoch_limit_minutes)}.")
            return (e.epoch_limit_minutes / e.estimated_training_time) / 1e5, 0, e.duration
        else:
            print(f"error '{nn}': failed to train. Error: {e}")
            if fail_iterations < 0:
                return accuracy_duration
            else:
                raise e


class Train:
    def __init__(self, config: tuple[str, str, str, str], out_shape: tuple, minimum_accuracy: float, batch: int, nn_module, task,
                 train_dataset, test_dataset, metric, num_workers, prm: dict, save_to_db=True, is_code=False,
                 tb_log_dir: str = "runs/experiment_1"):
        """
        Universal class for training CV, Text Generation and other models.
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
        self.metric_function = self.load_metric_function(metric)
        self.save_to_db = save_to_db
        self.is_code = is_code

        self.num_workers = num_workers
        self.train_loader = train_loader_f(self.train_dataset, self.batch, num_workers)
        self.test_loader = test_loader_f(self.test_dataset, self.batch, num_workers)

        self.in_shape = get_in_shape(train_dataset, num_workers)
        self.device = torch_device()

        # Load model
        model_net = get_attr(nn_module, 'Net')
        self.model_name = nn_module
        self.model = model_net(self.in_shape, out_shape, prm, self.device)
        self.model.to(self.device)

        # Initialize loss function for tracking
        self.loss_fn = self._get_loss_function()

        # Epoch metrics history
        self.epoch_history: List[EpochMetrics] = []
        self.best_accuracy = 0.0
        self.best_epoch = 0

        # ✅ TensorBoard run directory for THIS training
        self.tb_log_dir = _ensure_dir(tb_log_dir)

    def _get_loss_function(self):
        """Get loss function for metric tracking"""
        if hasattr(self.model, 'criterion'):
            return self.model.criterion
        elif hasattr(self.model, 'loss_fn'):
            return self.model.loss_fn
        else:
            if 'classification' in self.task or 'img-class' in self.task:
                return torch.nn.CrossEntropyLoss()
            elif 'segmentation' in self.task:
                return torch.nn.CrossEntropyLoss()
            else:
                return torch.nn.MSELoss()

    def _compute_loss(self, data_loader) -> float:
        """Compute average loss over a dataset"""
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

    def _compute_accuracy(self, data_loader) -> float:
        """Compute accuracy over a dataset using the metric function"""
        self.model.eval()
        self.metric_function.reset()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.metric_function(outputs, labels)

        return self.metric_function.result()

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
        """ Training and evaluation with comprehensive metrics tracking """

        start_time = time.time_ns()
        self.model.train_setup(self.prm)
        accuracy_to_time = 0.0
        duration = sys.maxsize

        optimizer = getattr(self.model, 'optimizer', None)

        # ✅ TensorBoard logging setup
        tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = SummaryWriter(log_dir=self.tb_log_dir)
        except ImportError:
            tb_writer = None

        for epoch in range(1, epoch_max + 1):
            epoch_start_time = time.time_ns()
            print(f"epoch {epoch}", flush=True)

            # Training phase
            self.model.train()
            self.model.learn(DataRoll(self.train_loader, epoch_limit_minutes))

            # Compute gradient norm after training
            grad_norm = compute_gradient_norm(self.model)

            # Get current learning rate
            lr_now = get_current_lr(optimizer)

            # Compute losses
            train_loss = self._compute_loss(self.train_loader)
            test_loss = self._compute_loss(self.test_loader)

            # Compute accuracies
            train_accuracy = self._compute_accuracy(self.train_loader)
            test_accuracy = self.eval(self.test_loader)

            accuracy = test_accuracy
            accuracy = 0.0 if math.isnan(accuracy) or math.isinf(accuracy) else accuracy
            duration = time.time_ns() - start_time
            epoch_duration = (time.time_ns() - epoch_start_time) / 1e9  # seconds

            # Calculate throughput
            total_samples = len(self.train_dataset)
            samples_per_second = total_samples / max(epoch_duration, 0.001)

            # Track best accuracy
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch

            # Record epoch metrics
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                train_accuracy=train_accuracy,
                test_accuracy=accuracy,
                lr=lr_now if lr_now is not None else 0.0,
                gradient_norm=grad_norm,
                samples_per_second=samples_per_second
            )
            self.epoch_history.append(epoch_metrics)

            # ✅ TensorBoard logging (scalars always)
            if tb_writer:
                tb_writer.add_scalar('Loss/train', train_loss, epoch)
                tb_writer.add_scalar('Loss/val', test_loss, epoch)
                tb_writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                tb_writer.add_scalar('Accuracy/val', accuracy, epoch)
                if lr_now is not None:
                    tb_writer.add_scalar('Learning_Rate', lr_now, epoch)
                tb_writer.add_scalar('Gradient_Norm', grad_norm, epoch)
                tb_writer.add_scalar('Throughput', samples_per_second, epoch)

                # ✅ LIVE age-estimation visualizations (ALWAYS, not only save_to_db/is_code/save_path)
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
                            # If outputs are logits, convert to age class
                            if outputs.ndim > 1 and outputs.shape[1] > 1:
                                preds = outputs.argmax(dim=1).detach().cpu().numpy()
                            else:
                                preds = outputs.squeeze().detach().cpu().numpy()

                            true = labels.squeeze().detach().cpu().numpy()
                            pred_list.extend(preds.tolist() if hasattr(preds, 'tolist') else preds)
                            true_list.extend(true.tolist() if hasattr(true, 'tolist') else true)

                            if len(img_list) < 8:
                                take = min(8 - len(img_list), inputs.shape[0])
                                img_list.extend(inputs[:take].detach().cpu())

                            if len(img_list) >= 8 and len(pred_list) >= 64:
                                break

                    import numpy as np
                    import matplotlib.pyplot as plt
                    pred_arr = np.array(pred_list)
                    true_arr = np.array(true_list)

                    # Scatter plot
                    if len(pred_arr) > 0 and len(true_arr) > 0:
                        fig = plt.figure(figsize=(5, 5))
                        plt.scatter(true_arr, pred_arr, alpha=0.5)
                        plt.xlabel('True Age')
                        plt.ylabel('Predicted Age')
                        plt.title('Predicted vs. True Age')
                        tb_writer.add_figure('Scatter/Pred_vs_True_Age', fig, epoch)
                        plt.close(fig)

                    # Histogram
                    if len(pred_arr) > 0:
                        fig_hist = plt.figure(figsize=(6, 3))
                        plt.hist(pred_arr, bins=20, alpha=0.7, label='Predicted')
                        plt.hist(true_arr, bins=20, alpha=0.5, label='True')
                        plt.legend()
                        plt.title('Age Distribution')
                        tb_writer.add_figure('Histogram/Age_Distribution', fig_hist, epoch)
                        plt.close(fig_hist)

                    # Images + labels
                    if len(img_list) > 0:
                        import torchvision
                        grid = torchvision.utils.make_grid(img_list, nrow=4, normalize=True)
                        tb_writer.add_image('Samples/Images', grid, epoch)

                        labels_text = [
                            f"P:{int(p)} T:{int(t)}"
                            for p, t in zip(pred_arr[:len(img_list)], true_arr[:len(img_list)])
                        ]

                        fig_img = plt.figure(figsize=(10, 3))
                        plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
                        plt.axis('off')
                        for idx, label in enumerate(labels_text):
                            x = 10 + (idx % 4) * 120
                            y = 20 + (idx // 4) * 120
                            plt.text(x, y, label, color='white', fontsize=8,
                                     bbox=dict(facecolor='black', alpha=0.5))
                        tb_writer.add_figure('Samples/Images_with_Labels', fig_img, epoch)
                        plt.close(fig_img)

                except Exception as viz_e:
                    print(f"[WARN] TensorBoard viz failed (epoch {epoch}): {viz_e}")

                # ✅ flush so Windows sees updates live
                tb_writer.flush()

            # Print detailed metrics
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {test_loss:.4f}")
            print(f"  Train Acc:  {train_accuracy:.4f} | Val Acc:  {accuracy:.4f}  [Best: {self.best_accuracy:.4f} @ epoch {self.best_epoch}]")
            if lr_now and grad_norm and samples_per_second:
                print(f"  LR: {lr_now:.6f} | Grad Norm: {grad_norm:.4f} | Throughput: {samples_per_second:.1f} samples/s")
            print(f"  Epoch time: {epoch_duration:.1f}s | Trial elapsed: {(time.time_ns() - start_time) / 1e9:.1f}s", flush=True)

            accuracy_to_time = accuracy_to_time_metric(accuracy, self.minimum_accuracy, duration)
            if not good(accuracy, self.minimum_accuracy, duration):
                raise AccuracyException(
                    accuracy, duration,
                    f"Accuracy is too low: {accuracy}."
                    f" The minimum accepted accuracy for the '{self.config[1]}' dataset is {self.minimum_accuracy}."
                )

            if save_pth_weights or save_onnx_weights:
                save_if_best(self.model, self.model_name, accuracy, save_pth_weights, save_onnx_weights,
                             train_set, self.num_workers, save_path=save_path)

            only_prm = {k: v for k, v in self.prm.items() if k not in {'uid', 'duration', 'accuracy', 'epoch'}}
            prm = merge_prm(self.prm, {
                'uid': uuid4(only_prm),
                'duration': duration,
                'accuracy': accuracy,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_accuracy,
                'gradient_norm': grad_norm,
                'samples_per_second': samples_per_second,
                'best_accuracy': self.best_accuracy,
                'best_epoch': self.best_epoch,
            } | ({'lr_now': lr_now} if lr_now else {})
              | ({'gpu_memory_kb': get_gpu_memory_kb()} if get_gpu_memory_kb else {}))

            if self.save_to_db:
                # Keep your DB save logic as-is
                if self.is_code:
                    if save_path:
                        save_results(self.config + (epoch,), join(save_path, f"{epoch}.json"), prm)
                    else:
                        print(f"[WARN]parameter `save_Path` set to null, the statics will not be saved into a file.")
                else:
                    if save_path is None:
                        save_path = model_stat_dir(self.config)
                    save_results(self.config + (epoch,), join(save_path, f"{epoch}.json"), prm)
                    DB_Write.save_results(self.config + (epoch,), prm)

        # ✅ close TensorBoard cleanly
        if tb_writer:
            tb_writer.close()

        # Save training summary at the end
        if save_path and self.epoch_history:
            self._save_training_summary()

        if hasattr(self.test_dataset, 'held_out_test'):
            test_loader = test_loader_f(self.test_dataset.held_out_test, self.batch, self.num_workers)
            self.model.eval()
            held_out_loss = self._compute_loss(test_loader)
            held_out_acc = self.eval(test_loader)
            print(f"\n{'='*60}")
            print(f"  HELD-OUT TEST SET (20%) — final result, never used during training")
            print(f"  Test Loss: {held_out_loss:.4f} | Test Acc: {held_out_acc:.4f} (MAE ~{(1.0 - held_out_acc) * 20.0:.2f} yrs)")
            print(f"{'='*60}", flush=True)

        total_trial_seconds = (time.time_ns() - start_time) / 1e9
        print(f"\n{'='*60}")
        print(f"  Trial complete | {epoch_max} epochs | Total time: {total_trial_seconds/60:.1f} min ({total_trial_seconds:.0f}s)")
        print(f"  Best Val Acc: {self.best_accuracy:.4f} @ epoch {self.best_epoch}")
        print(f"{'='*60}\n", flush=True)

        return accuracy, accuracy_to_time, duration

    def _save_training_summary(self):
        """Save comprehensive training summary"""
        import json
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
            'training_summary': {
                'total_epochs': len(self.epoch_history),
                'best_accuracy': self.best_accuracy,
                'best_epoch': self.best_epoch,
                'final_train_loss': self.epoch_history[-1].train_loss if self.epoch_history else 0,
                'final_test_loss': self.epoch_history[-1].test_loss if self.epoch_history else 0,
                'final_accuracy': self.epoch_history[-1].test_accuracy if self.epoch_history else 0,
                'gpu_memory_kb': get_gpu_memory_kb(),
            },
            'learning_curves': {
                'epochs': [e.epoch for e in self.epoch_history],
                'train_loss': [e.train_loss for e in self.epoch_history],
                'test_loss': [e.test_loss for e in self.epoch_history],
                'train_accuracy': [e.train_accuracy for e in self.epoch_history],
                'test_accuracy': [e.test_accuracy for e in self.epoch_history],
                'lr': [e.lr for e in self.epoch_history],
                'gradient_norm': [e.gradient_norm for e in self.epoch_history],
            },
            'epoch_details': [asdict(e) for e in self.epoch_history]
        }

        summary_path = out_dir / 'training_summary.json'
        try:
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"Training summary saved to {summary_path}")
        except Exception as e:
            print(f"[WARN] Failed to save training summary: {e}")

    def eval(self, test_loader):
        """Evaluation with standardized metric interface"""
        if debug:
            for inputs, labels in test_loader:
                print(f"[EVAL DEBUG] labels type: {type(labels)}")
                if isinstance(labels, torch.Tensor):
                    print(f"[EVAL DEBUG] labels shape: {labels.shape}")
                else:
                    print(f"[EVAL DEBUG] labels sample: {labels[:2]}")
        self.model.eval()

        self.metric_function.reset()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                self.metric_function(outputs, labels)

        return self.metric_function.result()


def train_new(nn_code, task, dataset, metric, prm, save_to_db=True, prefix: Union[str, None] = None,
              save_path: Union[str, None] = None, export_onnx=False,
              epoch_limit_minutes=default_epoch_limit_minutes, transform_dir=None):
    """
    train the model with the given code and hyperparameters and evaluate it.
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

        # ✅ unique TB dir for code-based run too
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