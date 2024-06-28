from pathlib import Path
import warnings

import numpy as np
from sklearn import metrics
import torch
from keras.callbacks import Callback


class CheckPointCallback(Callback):
    def __init__(
        self,
        save_path,
        map_location=None,
        **modules,
    ):
        super().__init__()
        self.save_path = Path(save_path)
        self.modules = modules
        self.map_location = map_location
        self.epoch = 0
        modules["ckpt"] = self

    def restore(self):
        try:
            checkpoint = torch.load(self.save_path, map_location=self.map_location)
        except FileNotFoundError as err:
            print("Initializing from scratch.")
            return

        for name, module in self.modules.items():
            if hasattr(module, "load_state_dict"):
                module.load_state_dict(checkpoint[name])
            else:
                self.modules[name] = checkpoint[name]

        print(f"Restored from {str(self.save_path)}")

    def save(self):
        to_be_saved = {
            name: module.state_dict() for name, module in self.modules.items()
        }
        torch.save(to_be_saved, self.save_path)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        self.save()

    def state_dict(self, prefix=""):
        return {f"{prefix}epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]


class BestCheckPointCallback(CheckPointCallback):
    def __init__(
        self,
        monitor,
        save_path,
        min_delta=0.0001,
        min_is_better=True,
        map_location=None,
        monitor_not_available_skip=False,
        **modules,
    ) -> None:
        self.monitor = monitor
        self.min_delta = min_delta
        self.min_is_better = min_is_better
        self.monitor_not_available_skip = monitor_not_available_skip
        self.best_score = np.inf

        super().__init__(
            save_path=save_path,
            map_location=map_location,
            **modules,
        )

    def state_dict(self, prefix=""):
        state_dict = super().state_dict(prefix=prefix)
        state_dict[f"{prefix}best_score"] = self.best_score
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict=state_dict)
        self.best_score = state_dict["best_score"]

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor not in logs and self.monitor_not_available_skip:
            warnings.warn(f"Monitor '{self.monitor}' is not available.")
            return

        var = logs[self.monitor]

        if not self.min_is_better:
            var = -var

        if var + self.min_delta < self.best_score:
            self.best_score = var
            super().on_epoch_end(logs=logs, epoch=epoch)


class EvaluationCallback(Callback):
    def __init__(self, val_loader, freq=1, prefix=None):
        super().__init__()
        self.val_loader = val_loader
        self.prefix = prefix
        self.freq = freq

        self.current_step = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            raise Exception("The given logs by the trainer is None")
        self.current_step += 1
        if self.current_step % self.freq == 0:
            self.log(logs)

    def log(self, logs):
        predictions = self.model.predict(self.val_loader, verbose=0)
        metric_logs = self.compute_metrics(**predictions)
        metric_logs = self.update_keys(metric_logs)
        logs.update(metric_logs)

    def update_keys(self, metric_logs):
        return (
            metric_logs
            if self.prefix is None
            else {f"{self.prefix}_{key}": value for key, value in metric_logs.items()}
        )

    def compute_metrics(self, y_true, y_pred, **kwds):
        y_score = y_pred
        y_pred = np.round(y_score)

        m = {}
        m["acc"] = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
        m["f1"] = metrics.f1_score(y_true=y_true, y_pred=y_pred)
        m["precision"] = metrics.f1_score(y_true=y_true, y_pred=y_pred)
        m["recall"] = metrics.f1_score(y_true=y_true, y_pred=y_pred)
        try:
            m["roc"] = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        except ValueError:
            pass

        try:
            precision, recall, _ = metrics.precision_recall_curve(
                y_true=y_true, probas_pred=y_score
            )
            m["pr"] = metrics.auc(recall, precision)
        except ValueError:
            pass

        return m
