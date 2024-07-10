import keras
import torch


class BaseTorchTrainer(keras.models.Model):
    def compile(
        self,
        optimizer,
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
        auto_scale_loss=True,
    ):
        super().compile(
            optimizer=None,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
        )
        self._torch_optimizer = optimizer

    def _compute_loss(self, data, y_pred):
        return self.loss(y_true=data["y_true"], y_pred=y_pred)

    @property
    def metrics_names(self):
        names = [m.name for m in self.metrics]
        for m in self.metrics:
            if hasattr(m, "additional_names"):
                names.extend(m.additional_names)

        return names

    @torch.inference_mode()
    def _compute_metrics(self, data, y_pred, loss=None):
        y_true = data["y_true"]
        logs = {}
        for metric in self.metrics:
            if metric.name == "loss":
                if loss is not None:
                    logs["loss"] = metric(loss)
            else:
                result = metric(y_pred=y_pred, y_true=y_true)
                if isinstance(result, dict):
                    logs.update(result)
                else:
                    logs[metric.name] = result
        return logs

    def _graph(self, data, training):
        y_pred = self.call(data, training=training)

        loss = self._compute_loss(data, y_pred)
        logs = self._compute_metrics(data, y_pred, loss)

        return logs, loss

    def train_step(self, data):
        logs, loss = self._graph(data, training=True)
        self._torch_optimizer.zero_grad()
        loss.backward()
        self._torch_optimizer.step()
        return logs

    @torch.inference_mode()
    def predict_step(self, data, training=False):
        outputs = {"y_pred": self.call(data, training=training)}
        outputs["y_true"] = data.get("y_true")
        return outputs

    @torch.inference_mode()
    def test_step(self, data):
        logs, loss = self._graph(data, training=False)
        return logs


class PatchTorchTrainer(BaseTorchTrainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self._in_sequence_mode = True

    def aggragte(self, x):
        return x.mean()

    def _aggragate_fn(self, x, lenghts):
        splited = x.split(lenghts, dim=0)
        return torch.stack([self.aggragte(x) for x in splited])

    @property
    def in_sequence_mode(self):
        return self._in_sequence_mode

    @in_sequence_mode.setter
    def in_sequence_mode(self, value):
        self._in_sequence_mode = value

    def predict_step(self, data, training=False):
        if not self.in_sequence_mode:
            return super().predict_step(data, training=training)
        predictions = super().predict_step(data, training=training)
        y_pred = self._aggragate_fn(
            predictions["y_pred"], lenghts=data["sample_length"]
        )
        return {"y_pred": y_pred, "y_true": data["y_true"]}

    def test_step(self, data):
        if not self.in_sequence_mode:
            return super().test_step(data)

        predictions = self.predict_step(data, training=False)
        logs = self._compute_metrics(data=data, y_pred=predictions["y_pred"])
        return logs
