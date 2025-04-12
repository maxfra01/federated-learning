import torch
from torch import nn


class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size=80, embed_dim=8, lstm_hidden_dim=256, seq_len=80, batch_size=32):
        super(ShakespeareLSTM, self).__init__()

        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lstm_hidden_dim = lstm_hidden_dim
        self.vocab_size = vocab_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(input_size=lstm_hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)

        # Dense output layer
        self.dense = nn.Linear(lstm_hidden_dim, vocab_size)

    def init_hidden(self, batch_size):
        """
        Inizializza lo stato nascosto e la cella della LSTM come tensori di zeri.
        """
        h0 = torch.zeros(2, batch_size, self.lstm_hidden_dim).to(self.device)
        c0 = torch.zeros(2, batch_size, self.lstm_hidden_dim).to(self.device)
        return (h0, c0)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        batch_size = x.size(0)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        x, hidden = self.lstm1(x, hidden)  # (batch_size, seq_len, lstm_hidden_dim)

        x, hidden = self.lstm2(x, hidden)  # (batch_size, seq_len, lstm_hidden_dim)

        x = self.dense(x)  # (batch_size, seq_len, vocab_size)

        return x, hidden