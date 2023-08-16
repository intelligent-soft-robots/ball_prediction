import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic data
def generate_data(num_samples):
    ball_angles = np.random.uniform(low=0, high=np.pi/2, size=num_samples)
    rebound_angles = np.random.uniform(low=0, high=np.pi/2, size=num_samples)
    return ball_angles, rebound_angles

# Define the physics-informed model with configurable layers
class ReboundModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neurons):
        super(ReboundModel, self).__init__()

        layers = []
        prev_dim = input_dim
        for num_neurons in hidden_neurons:
            layers.append(nn.Linear(prev_dim, num_neurons))
            layers.append(nn.ReLU())
            prev_dim = num_neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Loss function that enforces the physics constraint
def physics_loss(rebound_angles_pred, ball_angles):
    loss = torch.mean(torch.abs(rebound_angles_pred - ball_angles))
    return loss

# Training loop
def train_model(model, ball_angles, rebound_angles, num_epochs, learning_rate):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        ball_angles_tensor = torch.tensor(ball_angles, dtype=torch.float32).unsqueeze(1)
        rebound_angles_tensor = torch.tensor(rebound_angles, dtype=torch.float32).unsqueeze(1)

        rebound_angles_pred = model(ball_angles_tensor)

        loss = physics_loss(rebound_angles_pred, ball_angles_tensor)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Main function
if __name__ == '__main__':
    num_samples = 1000
    num_epochs = 100
    learning_rate = 0.001
    hidden_neurons = [64, 128]  # Number of neurons for each hidden layer

    ball_angles, rebound_angles = generate_data(num_samples)

    input_dim = 1  # Dimension of input data
    output_dim = 1  # Dimension of output data

    model = ReboundModel(input_dim, output_dim, hidden_neurons)
    train_model(model, ball_angles, rebound_angles, num_epochs, learning_rate)
