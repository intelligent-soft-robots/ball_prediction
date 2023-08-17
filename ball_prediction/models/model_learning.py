import os
from typing import Callable, List, Type, Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Kernel
from sklearn.metrics import mean_squared_error
from torch.utils.tensorboard import SummaryWriter


class ReboundModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_neurons: List[int],
        use_layer_norm: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        super(ReboundModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for num_neurons in hidden_neurons:
            layers.append(nn.Linear(prev_dim, num_neurons))
            if use_layer_norm:
                layers.append(nn.LayerNorm(num_neurons))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = num_neurons
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class BaseTraining:
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        learning_rate: float,
        use_tensorboard: bool,
        use_wandb: bool,
    ):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        if self.use_wandb:
            wandb.init()

    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        raise NotImplementedError("Subclasses should implement the train method.")

    def predict(self, inputs: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(inputs, dtype=torch.float32))
        return predictions.numpy()

    def save(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)

        training_state = {
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }
        state_path = os.path.join(save_dir, "training_state.pth")
        torch.save(training_state, state_path)

        if self.use_wandb:
            wandb.save(os.path.join(save_dir, "wandb.h5"))

        if self.use_tensorboard:
            self.writer.close()

    @classmethod
    def load(cls, load_dir: str, model_class: Type[nn.Module]):
        model_path = os.path.join(load_dir, "model.pth")
        model = model_class()  # Initialize an instance of the model class
        model.load_state_dict(torch.load(model_path))

        state_path = os.path.join(load_dir, "training_state.pth")
        training_state = torch.load(state_path)

        return cls(
            model,
            training_state["num_epochs"],
            training_state["learning_rate"],
            False,
            False,
        )  # Initialize without tensorboard and wandb


class DNNTraining(BaseTraining):
    def __init__(
        self,
        model: nn.Module,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: int = 10,
        lr_schedule_factor: float = 0.5,
        lr_schedule_patience: int = 5,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
    ) -> None:
        super().__init__(model, num_epochs, learning_rate, use_tensorboard, use_wandb)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_schedule_factor = lr_schedule_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.early_stopping_counter = 0
        self.best_loss = float("inf")

    def predict(self, input_data: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(input_data, dtype=torch.float32))
        return predictions.numpy()

    def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        if self.use_tensorboard:
            self.writer.add_graph(self.model, torch.tensor(inputs, dtype=torch.float32))

        if self.use_wandb:
            wandb.watch(self.model)

        optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.lr_schedule_factor,
            patience=self.lr_schedule_patience,
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            predicted = self.model(inputs)

            loss = self.mse_loss(predicted, targets)

            if self.use_wandb:
                wandb.log({"loss": loss})

            loss.backward()
            optimizer.step()

            lr_scheduler.step(loss)  # Update learning rate based on validation loss

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")

            # Early stopping
            if loss < self.best_loss:
                self.best_loss = loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            wandb.finish()


class PINNTraining(BaseTraining):
    def __init__(
        self,
        model: nn.Module,
        optimizer_class: Type[optim.Optimizer],
        optimizer_params: dict,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: int = 10,
        physics_loss_fn: Callable = None,  # Optional custom physics loss function
        use_tensorboard: bool = True,
        use_wandb: bool = True,
    ) -> None:
        super().__init__(model, num_epochs, learning_rate, use_tensorboard, use_wandb)

        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.weight_decay = weight_decay
        self.patience = patience
        self.early_stopping_counter = 0
        self.best_loss = float("inf")

        if physics_loss_fn is None:
            self.physics_loss_fn = self.default_physics_loss
        else:
            self.physics_loss_fn = physics_loss_fn

    def default_physics_loss(
        self, rebound_angles_pred: torch.Tensor, ball_angles: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.mean(torch.abs(rebound_angles_pred - ball_angles))
        return loss

    def predict(self, input_data: np.ndarray):
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(torch.tensor(input_data, dtype=torch.float32))
        return predictions.numpy()

    def train(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        if self.use_tensorboard:
            self.writer.add_graph(self.model, torch.tensor(inputs, dtype=torch.float32))

        if self.use_wandb:
            wandb.watch(self.model)

        optimizer = self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            ball_angles_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)

            output_pred = self.model(inputs)

            loss = self.physics_loss_fn(inputs, outputs, output_pred)

            if self.use_wandb:
                wandb.log({"loss": loss})

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}")

            # Early stopping
            if loss < self.best_loss:
                self.best_loss = loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        if self.use_tensorboard:
            self.writer.close()

        if self.use_wandb:
            wandb.finish()


class GaussianProcessTraining:
    def __init__(
        self,
        kernel: Kernel = None,
        kernel_params: dict = None,
        use_tensorboard: bool = True,
        use_wandb: bool = True,
    ):
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else {}
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        if self.kernel is None:
            self.kernel = 1.0 * RBF(length_scale=1.0)

        if self.use_tensorboard:
            self.writer = SummaryWriter()

        if self.use_wandb:
            wandb.init()

        self.model = GaussianProcessRegressor(kernel=self.kernel)

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        if self.use_tensorboard:
            self.writer.add_graph(
                torch.Tensor(), torch.tensor(inputs, dtype=torch.float32)
            )  # Dummy tensor

        if self.use_wandb:
            wandb.watch(self.model)

        self.model.fit(inputs, targets)

        if self.use_tensorboard:
            self.writer.close()

    def evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        predictions, _ = self.model.predict(inputs, return_std=True)
        loss = mean_squared_error(targets, predictions)

        if self.use_wandb:
            wandb.log({"loss": loss})

        return loss

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        predictions, _ = self.model.predict(inputs, return_std=True)
        return predictions

    def save(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "model.joblib")
        joblib.dump(self.model, model_path)

        kernel_path = os.path.join(save_dir, "kernel.npy")
        np.save(kernel_path, self.kernel)

        if self.use_wandb:
            wandb.save(os.path.join(save_dir, "wandb.h5"))

    @classmethod
    def load(cls, load_dir: str):
        model_path = os.path.join(load_dir, "model.joblib")
        model = joblib.load(model_path)

        kernel_path = os.path.join(load_dir, "kernel.npy")
        kernel = np.load(kernel_path, allow_pickle=True)

        return cls(
            kernel=kernel, use_tensorboard=False, use_wandb=False
        )  # Set wandb and tensorboard flags to False
