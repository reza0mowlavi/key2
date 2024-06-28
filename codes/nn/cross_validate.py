from pathlib import Path
import pickle
from typing import Sequence
from functools import wraps

import numpy as np
from sklearn import metrics

import torch
import keras
import tensorflow as tf

# import torchview
import optuna

from ._utils import Parallel
from .utils import LRSchedulerCallback, LinearScheduler, TimeoutCallback


class TimeoutException(Exception):
    pass


class NotEnoughPredictionsException(Exception):
    pass


def _clean_predictions(predictions):
    clean_predictions = []
    for pred in predictions:
        if isinstance(pred, Exception):
            print(pred)
        else:
            clean_predictions.append(pred)

    return clean_predictions, bool(len(predictions) == len(clean_predictions))


def _concat(predictions):
    def _chg_order(predictions):
        types = list(predictions[0])
        return {type: [pred[type] for pred in predictions] for type in types}

    def _concat_each(predictions):
        keys = predictions[0].keys()
        predictions = {key: [pred[key] for pred in predictions] for key in keys}
        return {key: np.concatenate(predictions[key]).flatten() for key in keys}

    predictions = _chg_order(predictions)

    return {key: _concat_each(predictions[key]) for key in predictions}


def _make_optimizer(parameters, hp_optimizer, n_iter_p_epoch):
    if hp_optimizer["use_weight_decay"]:
        optimizer = torch.optim.AdamW(
            params=parameters, weight_decay=hp_optimizer["weight_decay"]
        )
    else:
        optimizer = torch.optim.Adam(parameters)

    scheduler = LinearScheduler(
        optimizer=optimizer,
        initial_learning_rate=hp_optimizer["initial_learning_rate"],
        maximal_learning_rate=hp_optimizer["maximal_learning_rate"],
        left_steps=hp_optimizer["left_steps"] * n_iter_p_epoch,
        right_steps=hp_optimizer["right_steps"] * n_iter_p_epoch,
        decay_steps=hp_optimizer["decay_steps"] * n_iter_p_epoch + 1,
        minimal_learning_rate=hp_optimizer["minimal_learning_rate"],
    )

    return optimizer, scheduler


class ModelPipeline:
    def __init__(
        self,
        save_dir,
        tensorboard_dir,
        device="cpu",
        dtype=torch.float32,
        timeout=None,
        timeout_threshold=None,
    ):
        self.save_dir = Path(save_dir)
        self.tensorboard_dir = tensorboard_dir
        self.device = device
        self.dtype = dtype
        self.timeout = timeout
        self.timeout_threshold = timeout_threshold

    def extract_train_loader(self, train_loaders):
        raise NotImplementedError

    def make_trainer(
        self,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
    ):
        raise NotImplementedError

    def make_optimizer(
        self,
        trainer,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
    ):
        optimizer, scheduler = _make_optimizer(
            parameters=(p for p in trainer.parameters() if p.requires_grad),
            hp_optimizer=hp_optimizer,
            n_iter_p_epoch=len(self.extract_train_loader(train_loaders)),
        )
        return optimizer, scheduler

    def compile_trainer(
        self,
        trainer,
        optimizer,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
    ):
        raise NotImplementedError

    def make_callbacks(
        self,
        trainer,
        optimizer,
        scheduler,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
    ):
        callbacks = [LRSchedulerCallback(scheduler)]
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(self.tensorboard_dir / run_name / str(cv_number))
            )
        )
        # callbacks.append(
        # keras.callbacks.EarlyStopping(
        #    monitor="loss",
        #    start_from_epoch=hp_optimizer["epochs"] // 2,
        #    patience=10,
        #    mode="min",
        #    restore_best_weights=True,
        # )
        # )
        if self.timeout is not None:
            callbacks.append(TimeoutCallback(self.timeout))

        return callbacks

    def save_model(
        self,
        trainer,
        optimizer,
        scheduler,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
        callbacks,
    ):
        self.save_dir.mkdir(exist_ok=True, parents=False)
        (self.save_dir / "ckpts" / f"{run_name}").mkdir(exist_ok=True, parents=True)
        with open(
            self.save_dir / "ckpts" / f"{run_name}" / f"ckpt_{cv_number}.pickle", "wb"
        ) as f:
            torch.save(trainer.state_dict(), f)

    def fit_trainer(
        self,
        trainer,
        optimizer,
        scheduler,
        config,
        hp_optimizer,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
        callbacks,
    ):
        history = trainer.fit(
            x=self.extract_train_loader(train_loaders),
            epochs=hp_optimizer["epochs"],
            callbacks=callbacks,
            verbose=0,
        )
        return history

    def make_predictions(
        self,
        trainer,
        config,
        train_loaders,
        val_loaders,
        run_name,
        cv_number,
    ):
        return {
            key: trainer.predict(val_loaders[key], verbose=0) for key in val_loaders
        }

    def save_predictions(self, predictions, run_name, cv_number, history):
        self.save_dir.mkdir(exist_ok=True, parents=False)
        (self.save_dir / "preds" / f"{run_name}").mkdir(exist_ok=True, parents=True)
        with open(
            self.save_dir / "preds" / f"{run_name}" / f"pred_{cv_number}.pickle", "wb"
        ) as f:
            pickle.dump(predictions, f)

    def should_return_the_predictions(self, trainer, callbacks, hp_optimizer):
        if self.timeout is None:
            return

        timeout_callback = [x for x in callbacks if isinstance(x, TimeoutCallback)][0]
        if not timeout_callback.timeout_reached:
            return

        if self.timeout_threshold is not None and (
            timeout_callback.passed_epochs
            >= hp_optimizer["epochs"] * self.timeout_threshold
        ):
            return

        print("Raise timeout exception")
        raise TimeoutException

    def __call__(
        self, config, hp_optimizer, train_loaders, val_loaders, run_name, cv_number
    ):
        trainer = self.make_trainer(
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
        )
        optimizer, scheduler = self.make_optimizer(
            trainer=trainer,
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
        )
        self.compile_trainer(
            trainer=trainer,
            optimizer=optimizer,
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
        )
        callbacks = self.make_callbacks(
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
        )
        history = self.fit_trainer(
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
            callbacks=callbacks,
        )
        self.should_return_the_predictions(
            trainer=trainer,
            callbacks=callbacks,
            hp_optimizer=hp_optimizer,
        )  ### Cause an TimeoutException if needed
        predictions = self.make_predictions(
            trainer=trainer,
            config=config,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
        )
        self.save_predictions(
            predictions=predictions,
            history=history,
            run_name=run_name,
            cv_number=cv_number,
        )

        self.save_model(
            trainer=trainer,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            hp_optimizer=hp_optimizer,
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            run_name=run_name,
            cv_number=cv_number,
            callbacks=callbacks,
        )

        return predictions


class Objective:
    def __init__(
        self,
        save_dir,
        build_hp,
        make_model,
        dataset,
        objective_keys,
        tensorboard_dir,
        success_threshold=None,
        n_jobs=None,
        n_exp=None,
    ):
        self.save_dir = Path(save_dir)
        self.build_hp = build_hp
        self.make_model = make_model
        self.dataset = dataset
        self.objective_keys = objective_keys
        self.n_jobs = n_jobs
        self.tensorboard_dir = tensorboard_dir
        self.success_threshold = success_threshold
        self.n_exp = n_exp if n_exp is not None else len(dataset)

    def compute_metrics(self, y_true, y_pred):
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

    def log(self, trial, predictions):
        predictions = _concat(predictions)
        computed_metric = {
            f"{key}_{met_name}": met_val
            for key in predictions
            for met_name, met_val in self.compute_metrics(**predictions[key]).items()
        }

        run_name = "trial-%d" % trial.number
        if self.tensorboard_dir is not None:
            run_dir = (
                Path(self.tensorboard_dir) / run_name
            )  # This is the dir used in tensorboard_callback of optuna
            with tf.summary.create_file_writer(str(run_dir)).as_default():
                for name, value in computed_metric.items():
                    tf.summary.scalar(name, value, step=trial.number)
                    trial.set_user_attr(name, value)

        with open(self.save_dir / f"preds_{run_name}.pickle", "wb") as f:
            pickle.dump(predictions, f)

        return computed_metric

    def __call__(self, trial):
        config, hp_optimizer = self.build_hp(trial)
        predictions = Parallel(func=self.make_model, n_jobs=self.n_jobs)(
            dict(
                config=config,
                hp_optimizer=hp_optimizer,
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                run_name=f"trial-{int(trial.number)}",
                cv_number=cv_number,
            )
            for cv_number, (train_loaders, val_loaders) in zip(
                range(self.n_exp), self.dataset
            )
        )
        clean_predictions, was_clean = _clean_predictions(predictions)

        if len(clean_predictions) == 0:
            print("Not Enough Predictions")
            raise NotEnoughPredictionsException

        computed_metric = self.log(trial, clean_predictions)

        if not was_clean and (
            self.success_threshold is None
            or len(clean_predictions) < self.success_threshold
        ):
            print("Not Enough Predictions")
            raise NotEnoughPredictionsException

        return self._extract_objective_metric(computed_metric)

    def _extract_objective_metric(self, computed_metric):
        if isinstance(self.objective_keys, str):
            return computed_metric[self.objective_keys]

        result = [computed_metric[key] for key in self.objective_keys]
        result = result if len(result) > 1 else result[0]
        return result


class Optimizer:
    def __init__(
        self,
        objective,
        n_trials,
        n_startup_trials,
        study_name,
        save_dir,
        tensorboard_dir=None,
        timeout=None,
    ):
        self.objective = objective
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.study_name = study_name
        self.save_dir = Path(save_dir)
        self.tensorboard_dir = tensorboard_dir
        self.timeout = timeout
        self.sampler = None
        self.study = None

    def make_sampler(self):
        if (self.save_dir / "sampler.pickle").exists():
            sampler = pickle.load(open(self.save_dir / "sampler.pickle", "rb"))
            print("Sampler is loaded.")
        else:
            sampler = optuna.samplers.TPESampler(
                multivariate=True,
                n_startup_trials=self.n_startup_trials,
            )

        return sampler

    def objective_fn(self, trial):
        try:
            result = self.objective(trial)
            return result
        finally:
            with open(self.save_dir / "sampler.pickle", "wb") as f:
                pickle.dump(self.sampler, f)

    def make_study(self, sampler=None):
        storage_name = f"sqlite:///{self.save_dir}/{self.study_name}.db"
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
            sampler=sampler,
        )
        return study

    def make_callbacks(self):
        if self.tensorboard_dir is not None:
            return [
                optuna.integration.TensorBoardCallback(
                    self.tensorboard_dir, "objective"
                )
            ]

    def optimize(self):
        self.sampler = self.make_sampler()
        self.study = self.make_study(sampler=self.sampler)
        callbacks = self.make_callbacks()
        self.study.optimize(
            self.objective_fn,
            n_trials=(
                self.n_trials
                - len(self.study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
            ),
            callbacks=callbacks,
            timeout=self.timeout,
            catch=(TimeoutException, NotEnoughPredictionsException),
        )
