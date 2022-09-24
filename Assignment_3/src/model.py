import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, embedding_dim : int, hidden_size : int, batch_size : int, vocab_size : int, num_layers : int):
    """
    The function takes in the embedding dimension, hidden size, batch size, vocabulary size, and
    number of layers, and returns an embedding layer, an LSTM layer, and a fully connected layer
    
    :param embedding_dim: The size of the embedding vector
    :type embedding_dim: int
    :param hidden_size: The number of features in the hidden state h
    :type hidden_size: int
    :param batch_size: The number of sequences to pass through the network at a time
    :type batch_size: int
    :param vocab_size: The number of unique words in the vocabulary
    :type vocab_size: int
    :param num_layers: The number of layers in the LSTM
    :type num_layers: int
    """
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
    """
    The function takes in a batch of input words, passes them through an embedding layer, passes them
    through an LSTM, and then passes the output of the LSTM through a fully connected layer
    
    :param x: the input data
    :param h: hidden state
    :return: The output of the LSTM and the hidden state.
    """
    x = self.embedding(x)
    x, h = self.lstm(x,h)
    x = x.view(-1, self.embedding_dim)
    x = self.fc(x)
    return x,h






