import time

import numpy as np
import torch
import keras


class LinearScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate,
        maximal_learning_rate,
        left_steps,
        right_steps,
        decay_steps,
        minimal_learning_rate,
    ) -> None:
        self._initial_learning_rate = initial_learning_rate
        self._maximal_learning_rate = maximal_learning_rate
        self._left_steps = left_steps
        self._right_steps = right_steps
        self._decay_steps = decay_steps
        self._lowest_learning_rate = minimal_learning_rate

        left = np.linspace(initial_learning_rate, maximal_learning_rate, left_steps)
        right = np.linspace(maximal_learning_rate, initial_learning_rate, right_steps)
        decay = np.linspace(initial_learning_rate, minimal_learning_rate, decay_steps)
        self._lr = np.concatenate((left, right, decay))
        self.cycle_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> float:
        lr = (
            self._lr[self.cycle_step]
            if self.cycle_step < len(self._lr)
            else self._lr[-1]
        )
        self.cycle_step = (
            self.cycle_step + 1 if self.cycle_step < len(self._lr) else self.cycle_step
        )
        return [lr]


class CosineLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_learning_rate,
        maximal_learning_rate,
        left_steps,
        right_steps,
        decay_steps,
        minimal_learning_rate,
    ) -> None:
        self.initial_learning_rate = initial_learning_rate
        self.maximal_learning_rate = maximal_learning_rate
        self.left_steps = left_steps
        self.right_steps = right_steps
        self.decay_steps = decay_steps
        self.lowest_learning_rate = minimal_learning_rate

        left = (np.cos(np.linspace(np.pi, 2 * np.pi, left_steps)) + 1) * 0.5 * (
            maximal_learning_rate - initial_learning_rate
        ) + initial_learning_rate

        right = (np.cos(np.linspace(0, np.pi, right_steps)) + 1) * 0.5 * (
            maximal_learning_rate - minimal_learning_rate
        ) + initial_learning_rate

        decay = np.linspace(initial_learning_rate, minimal_learning_rate, decay_steps)

        self.lrs = np.concatenate((left, right, decay))
        self.cycle_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> float:
        lr = (
            self.lrs[self.cycle_step]
            if self.cycle_step < len(self.lrs)
            else self.lrs[-1]
        )
        self.cycle_step = (
            self.cycle_step + 1 if self.cycle_step < len(self.lrs) else self.cycle_step
        )
        return [lr]


class CosineAnnealingWarmstart(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        maximal_lr,
        minimal_lr,
        steps,
        warmup_steps,
        initial_lr,
    ):
        self.initial_lr = initial_lr
        self.maximal_lr = maximal_lr
        self.minimal_lr = minimal_lr
        self.warmup_steps = warmup_steps
        self.steps = steps

        if warmup_steps is not None:
            warmup_lr = np.linspace(initial_lr, maximal_lr, warmup_steps)

        cos_lr = (
            (np.cos(np.linspace(0, np.pi, steps)) + 1) * 0.5 * (maximal_lr - minimal_lr)
        ) + minimal_lr

        self.lrs = (
            cos_lr if self.warmup_steps is None else np.concatenate((warmup_lr, cos_lr))
        )

        self.cycle_step = 0
        super().__init__(optimizer)

    def get_lr(self) -> float:
        lr = (
            self.lrs[self.cycle_step]
            if self.cycle_step < len(self.lrs)
            else self.lrs[-1]
        )
        self.cycle_step = (
            self.cycle_step + 1 if self.cycle_step < len(self.lrs) else self.cycle_step
        )
        return [lr]


class LRSchedulerCallback(keras.callbacks.Callback):
    def __init__(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler, update_level="batch"
    ):
        super().__init__()
        if update_level not in ["epoch", "batch"]:
            raise ValueError(
                f'"update_level(={update_level})" must be either "batch" or "epoch".'
            )
        self.scheduler = scheduler
        self.update_level = update_level

    def step(self, logs=None):
        self.scheduler.step()
        if logs is not None:
            logs["lr"] = self.scheduler.optimizer.param_groups[0]["lr"]

    def on_train_batch_end(self, batch, logs=None):
        if self.update_level == "batch":
            self.step(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs["lr"] = self.scheduler.optimizer.param_groups[0]["lr"]
        if self.update_level == "epoch":
            self.step(logs)


class TimeoutCallback(keras.callbacks.Callback):
    def __init__(self, timeout):
        super().__init__()
        self._timeout = timeout
        self._start_time = None

        self.timeout_reached = False
        self.passed_steps = 0
        self.passed_epochs = 0

    def on_train_begin(self, logs=None):
        self._start_time = time.perf_counter()

    def on_batch_end(self, batch, logs=None):
        self.passed_steps += 1
        now = time.perf_counter()
        if now - self._start_time >= self._timeout:
            self.model.stop_training = True
            self.timeout_reached = True

    def on_epoch_end(self, epoch, logs=None):
        self.passed_epochs = epoch
