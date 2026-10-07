"""Causal residual TCN architecture and fitted-model wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import TCNConfig, TCNSpec
from .data import SensorScaler, fit_scaler


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TCN comparison requires optional PyTorch; install a CPU/GPU build compatible with the host"
        ) from error
    return torch, nn, F


def build_model(*, n_sensors: int, config: TCNConfig) -> Any:
    """Create a four-block causal residual TCN with a last-step linear head."""
    torch, nn, F = _torch()

    class CausalConv1d(nn.Conv1d):
        def __init__(self, channels_in: int, channels_out: int, dilation: int):
            super().__init__(channels_in, channels_out, config.kernel_size,
                             padding=0, dilation=dilation)
            self.left_padding = (config.kernel_size - 1) * dilation

        def forward(self, x):  # noqa: D401 - PyTorch API override
            return super().forward(F.pad(x, (self.left_padding, 0)))

    class ResidualBlock(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv1 = CausalConv1d(config.channels, config.channels, dilation)
            self.conv2 = CausalConv1d(config.channels, config.channels, dilation)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x):
            residual = x
            y = self.dropout(self.activation(self.conv1(x)))
            y = self.dropout(self.activation(self.conv2(y)))
            return self.activation(y + residual)

    class CausalResidualTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = CausalConv1d(n_sensors, config.channels, 1)
            self.blocks = nn.ModuleList(ResidualBlock(d) for d in config.dilations)
            self.head = nn.Linear(config.channels, 1)

        def forward(self, x):
            y = self.input_projection(x)
            for block in self.blocks:
                y = block(y)
            return self.head(y[:, :, -1]).squeeze(-1)

    return CausalResidualTCN()


@dataclass
class FittedTCN:
    """Trained TCN plus scaler and immutable candidate metadata."""

    spec: TCNSpec
    config: TCNConfig
    scaler: SensorScaler
    model: Any
    state_dict: dict[str, Any]

    def predict(self, values: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        """Predict finite RUL values without accessing labels."""
        torch, _, _ = _torch()
        transformed = self.scaler.transform(values)
        self.model.eval()
        batch_size = batch_size or self.config.batch_size
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(transformed), batch_size):
                tensor = torch.from_numpy(transformed[start:start + batch_size])
                outputs.append(self.model(tensor).detach().cpu().numpy())
        prediction = np.concatenate(outputs).astype("float64", copy=False)
        if len(prediction) != len(values) or not np.isfinite(prediction).all():
            raise ValueError("TCN predictions are nonfinite or malformed")
        return prediction


def fit_model(
    values: np.ndarray,
    metadata,
    spec: TCNSpec,
    config: TCNConfig,
    fit_engine_ids: set[int],
) -> FittedTCN:
    """Fit one candidate using engine-total-one weighted MSE."""
    torch, _, _ = _torch()
    if not fit_engine_ids:
        raise ValueError("fit_engine_ids cannot be empty")
    mask = metadata["engine_id"].isin(fit_engine_ids).to_numpy()
    if not mask.any():
        raise ValueError("fit engines are absent")
    raw_y = metadata.loc[mask, "rul"].to_numpy(dtype="float32")
    target = np.minimum(raw_y, spec.target_cap).astype("float32") if spec.target_cap else raw_y
    train_values = values[mask]
    counts = metadata.loc[mask].groupby("engine_id", sort=False)["engine_id"].transform("size").to_numpy()
    weights = (1.0 / counts).astype("float32")
    scaler = fit_scaler(values, metadata, fit_engine_ids)
    tensor_x = torch.from_numpy(scaler.transform(train_values))
    tensor_y = torch.from_numpy(target)
    tensor_w = torch.from_numpy(weights)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available() and config.fit_threads == 0:  # defensive; CPU is protocol default
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(config.fit_threads)
    model = build_model(n_sensors=values.shape[1], config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y, tensor_w)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
                                         shuffle=True, generator=generator,
                                         num_workers=0)
    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y, batch_w in loader:
            optimizer.zero_grad(set_to_none=True)
            error = model(batch_x) - batch_y
            loss = (error.square() * batch_w).sum() / batch_w.sum()
            loss.backward()
            optimizer.step()
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return FittedTCN(spec, config, scaler, model, state)


__all__ = ["FittedTCN", "build_model", "fit_model"]
