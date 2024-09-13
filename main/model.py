import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class weighted_visibility_graph(Dataset):
    def __init__(self, data, lookback, forecast, scaler=None):
        self.lookback = lookback
        self.forecast = forecast
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.data = self.scaler.fit_transform(data)
        self.num_nodes, self.num_steps = self.data.shape

    def __getitem__(self, index):
        start_index = np.random.randint(self.lookback, self.num_steps - self.forecast)
        end_index = start_index + self.forecast
        x = self._get_input_data(start_index)
        y = self._get_output_data(end_index)
        return x, y

    def _get_input_data(self, start_index):
        graph = self._construct_weighted_graph(start_index)
        return torch.from_numpy(nx.to_numpy_matrix(graph)).float()

    def _get_output_data(self, end_index):
        return torch.from_numpy(self.data[:, end_index - self.forecast:end_index]).float()

    def _construct_weighted_graph(self, start_index):
        data_subset = self.data[:, start_index - self.lookback:start_index]
        similarity_matrix = self._compute_similarity_matrix(data_subset)
        graph = nx.from_numpy_matrix(similarity_matrix)
        return graph

    def _compute_similarity_matrix(self, data_subset):
        euclidean_distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                distance = np.sqrt(np.sum((data_subset[i] - data_subset[j])**2))
                euclidean_distances[i, j] = distance
                euclidean_distances[j, i] = distance
        similarity_matrix = np.exp(-euclidean_distances / np.median(euclidean_distances))
        return similarity_matrix


class TransformerModel(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.input_proj = nn.Linear(num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x, y):
        x = self.input_proj(x)
        y = self.input_proj(y)
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        encoder_output = self.encoder(x)
        decoder_output = self.decoder(y, encoder_output)
        decoder_output = decoder_output.transpose(0, 1)
        output = self.output_proj(decoder_output)
        return output


# training
def train(model, optimizer, criterion, dataloader, device):
    model.train()
    total_loss = 0.0
    for i, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(x, y[:, :-1])
        loss = criterion(output, y[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# testing
def test(model, criterion, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            output = model(x, y[:, :-1])
            loss = criterion(output, y[:, 1:])
            total_loss += loss.item()
    return total_loss / len(dataloader)


# hyperparameter
num_nodes = 10
input_dim = 10
output_dim = 10
d_model = 64
nhead = 4
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
dropout = 0.1
batch_size = 64
lookback = 24
forecast = 24
lr = 1e-3
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 
data = np.random.randn(num_nodes, 1000)
dataset = weighted _visibility_graph(data, lookback, forecast)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = TransformerModel(num_nodes, input_dim, output_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
    
for epoch in range(num_epochs):
    train_loss = train(model, optimizer, criterion, dataloader, device)
    test_loss = test(model, criterion, dataloader, device)
    print(f"Epoch {epoch+1}, train_loss: {train_loss:.4f}, test_loss: {test_loss:.4f}")

print("Training completed.")

def predict(model, data, input_window, output_window, device):
    model.eval()
    with torch.no_grad():
        data = torch.FloatTensor(data).unsqueeze(0).to(device)
        seq_len, n_features = data.shape
        enc_output = model.encoder(data)
        predictions = torch.zeros(output_window, n_features).to(device)

        for i in range(output_window):
            input_seq = enc_output[:, -input_window:]
            output_seq = predictions[:i]
            if i > 0:
                input_seq = torch.cat([input_seq, output_seq], dim=1)
            pred = model.decoder(input_seq)
            predictions[i] = pred[-1]
    mse_loss = F.mse_loss(predictions, data[:, -output_window:])

    return predictions.cpu().numpy(), mse_loss.item()


def plot_prediction(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_true, label='true', linewidth=2)
    ax.plot(y_pred, label='predicted', linewidth=2)
    ax.legend(loc='upper left')
    ax.set(title=title, xlabel='time', ylabel='value')
    plt.show()


