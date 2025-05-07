
import torch
import torch.nn as nn
import gc
torch.cuda.empty_cache()
gc.collect()
# BI LSTM
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.view(N, value_len, self.heads, self.head_dim)
        keys = keys.view(N, key_len, self.heads, self.head_dim)
        queries = query.view(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention_weights = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        attention_weights = self.dropout(attention_weights)

        out = torch.einsum("nhql,nlhd->nqhd", [attention_weights, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        out = self.layer_norm(out + query)  # Residual connection with Layer Norm
        return out, attention_weights


# MultiLayerAttention class
class MultiLayerAttention(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout_rate=0.1):
        super(MultiLayerAttention, self).__init__()
        self.layers = nn.ModuleList([MultiHeadAttention(embed_size, heads, dropout_rate) for _ in range(num_layers)])

    def forward(self, values, keys, query, mask=None):
        out = query
        attention_weights_list = []
        for layer in self.layers:
            out, attention_weights = layer(values, keys, out, mask)
            attention_weights_list.append(attention_weights)
        attention_weights = torch.stack(attention_weights_list, dim=0).mean(dim=0)
        return out, attention_weights


# EnhancedBiLSTMWithAttention model 
class EnhancedBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, num_classes, dropout_rate=0.1):
        super(EnhancedBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM instead of BiGRU
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, 
                          batch_first=True, dropout=dropout_rate)
        self.lstm_layer_norm = nn.LayerNorm(hidden_size * 2)  # Normalize LSTM output

        # Multi-layer attention mechanism
        self.attention = MultiLayerAttention(embed_size=hidden_size * 2, heads=num_heads, num_layers=3, dropout_rate=dropout_rate)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM output - now requires both h0 and c0
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.lstm_layer_norm(lstm_out)  # Layer normalization

        # Apply multi-head attention
        out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Global average pooling
        out = out.mean(dim=1)

        out = self.dropout(out)
        out = self.fc(out)
        return out, attention_weights