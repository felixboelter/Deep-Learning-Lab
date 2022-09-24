import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, embedding_dim : int, hidden_size : int, batch_size : int, vocab_size : int, num_layers : int):
        super(LSTMModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers)
        self.fc = nn.Linear(in_features=embedding_dim,out_features=vocab_size)

  def forward(self, x,h):
        x = self.embedding(x)
        x, h = self.lstm(x,h)
        x = x.view(-1, self.embedding_dim)
        x = self.fc(x)
        
        return x,h






