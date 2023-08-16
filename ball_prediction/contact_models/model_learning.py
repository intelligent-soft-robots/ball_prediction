import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


class BaseTraining:
    def __init__(self, model, num_epochs, learning_rate):
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def save(self, save_dir):
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

    @classmethod
    def load(cls, load_dir, model_class):
        model_path = os.path.join(load_dir, "model.pth")
        model = model_class()  # Initialize an instance of the model class
        model.load_state_dict(torch.load(model_path))

        state_path = os.path.join(load_dir, "training_state.pth")
        training_state = torch.load(state_path)

        return cls(model, training_state["num_epochs"], training_state["learning_rate"])


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


class PINNTraining(BaseTraining):
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: int = 10,
    ) -> None:
        super().__init__(model, num_epochs, learning_rate)

        self.weight_decay = weight_decay
        self.patience = patience
        self.early_stopping_counter = 0
        self.best_loss = float("inf")

    def physics_loss(
        self, rebound_angles_pred: torch.Tensor, ball_angles: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.mean(torch.abs(rebound_angles_pred - ball_angles))
        return loss

    def train(self, ball_angles: torch.Tensor) -> None:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            ball_angles_tensor = torch.tensor(
                ball_angles, dtype=torch.float32
            ).unsqueeze(1)

            rebound_angles_pred = self.model(ball_angles_tensor)

            loss = self.physics_loss(rebound_angles_pred, ball_angles_tensor)

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


class DNNTraining(BaseTraining):
    def __init__(
        self,
        model: nn.Module,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float = 0.0,
        patience: int = 10,
        lr_schedule_factor: float = 0.5,
        lr_schedule_patience: int = 5,
    ) -> None:
        super().__init__(model, num_epochs, learning_rate)

        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_schedule_factor = lr_schedule_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.early_stopping_counter = 0
        self.best_loss = float("inf")

    def train(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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


# Main function
if __name__ == "__main__":
    num_samples = 1000
    num_epochs = 100
    learning_rate = 0.001
    hidden_neurons = [64, 128]  # Number of neurons for each hidden layer
    use_layer_norm = True  # Enable or disable layer normalization
    dropout_rate = 0.2  # Dropout rate, set to 0 to disable dropout

    inputs = torch.randn(num_samples, input_dim)  # Replace with your actual input data
    targets = torch.randn(
        num_samples, output_dim
    )  # Replace with your actual target data

    input_dim = inputs.shape[1]  # Dimension of input data
    output_dim = targets.shape[1]  # Dimension of output data

    model = MLP(input_dim, output_dim, hidden_neurons, use_layer_norm, dropout_rate)
    trainer = DNNTraining(model, num_epochs, learning_rate)
    trainer.train(inputs, targets)
