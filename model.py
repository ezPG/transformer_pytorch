import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class InputEmbeddings(nn.Module):
    """
    INPUT EMBEDDING MODULE
    
    This module is responsible for embedding the input tokens into a dense vector representation.
    It takes the vocabulary size and the embedding dimension as input and creates an embedding matrix
    to map each token to its corresponding vector representation.
    """
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        
        self.embed_size = torch.tensor(embed_size)
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
    def forward(self, x):
        
        return self.embedding(x) * torch.sqrt(self.embed_size)
        

class PositionalEncoding(nn.Module):
    """
    COSINE POSITION ENCODING MODULE
    
    This module is responsible for encoding the positional information of the input sequence.
    It generates a matrix of positional encodings using sine and cosine functions, which is added
    to the input embeddings to make the model aware of the token positions in the sequence.
    """
    def __init__(self, embed_size, seq_len, dropout = 0.1):
        super().__init__()

        self.embed_size = embed_size
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p = dropout)
        
        self.pos_enc = torch.zeros(seq_len, embed_size)
        
        self.position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) ##shape: (seq_len, 1)
        
        self.denominator = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0, dtype=torch.float))/embed_size))  ##exp with ln for numerical stability
    
        self.pos_enc[:, 0::2] = torch.sin(self.position / self.denominator) ##Every even position values from 0
        self.pos_enc[:, 1::2] = torch.cos(self.position / self.denominator) ##Every odd position values from 1
        
        self.pos_enc = self.pos_enc.unsqueeze(0) #(1, seq_len, emed_dim)
        
        # self.register_buffer('pos_enc', self.pos_enc) ##save this encoding along with the saved model, for inference.
    
    def forward(self, x):
        
        x = x + self.pos_enc[:, :x.shape[1], :].to(x.device).requires_grad_(False)
        
        return self.dropout((x))
        
        
class LayerNorm(nn.Module):
    """
    LAYER NORMALIZATION MODULE
    
    This module performs Layer Normalization, which is a technique used to stabilize the training
    of deep neural networks. It normalizes the activations across the features (or channels) within
    a single data sample, instead of across the batch like Batch Normalization.
    """
    def __init__(self, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1)) ##Multiplicative Parameter
        self.beta = nn.Parameter(torch.zeros(1)) ##Additive Parameter (BIAS)
        
    def forward(self, x):
        
        mean = torch.mean(x, dim = -1, keepdims = True)
        std = torch.std(x, dim = -1, keepdims = True)
        
        return self.gamma * ((x - mean)/(std + self.eps)) + self.beta


class FeedForward(nn.Module):
    """
    FEED-FORWARD NETWORK MODULE
    
    This module is a simple feed-forward neural network with two linear layers and a ReLU activation
    in between. It is used as a part of the encoder and decoder blocks in the Transformer architecture
    to apply non-linear transformations to the input.
    """
    def __init__(self, embed_size, d_ff, dropout = 0.5):
        super().__init__()
    
        self.fc1 = nn.Linear(embed_size, d_ff, bias = True) ##W1, B1
        self.dropout = nn.Dropout(p=dropout)
        
        self.fc2 = nn.Linear(d_ff, embed_size, bias = True) ##W2, B2
        self.activation = nn.ReLU()
        
    def forward(self, x):

        a = self.activation(self.fc1(x))
        
        return self.fc2(self.dropout(a))
        

# class Attention(nn.Module):
#     """
#     ATTENTION MODULE
    
#     This class is a placeholder for the Attention module, which is a crucial component of the Transformer
#     architecture. It computes the attention weights between the query, key, and value vectors, and applies
#     a weighted sum of the value vectors based on the computed attention scores.
#     """
#     def __init__(self):
#         super().__init__()
#     def forward():
#         return

class MultiHeadAttention(nn.Module):
    """
    MULTI-HEAD ATTENTION MODULE
    
    This module implements the Multi-Head Attention mechanism, which is a key component of the Transformer
    architecture. It splits the input into multiple heads, computes the attention independently for each
    head, and then concatenates the results. This allows the model to attend to different parts of the
    input sequence in parallel.
    """
    def __init__(self, embed_size, nheads, dropout = 0.1):
        super().__init__()
        
        assert embed_size % nheads == 0, f"embed_size: {embed_size} (d_model) is not divisible by nheads: {nheads}"
        
        self.embed_size = embed_size
        self.nheads = nheads
        self.dk = embed_size // nheads
        
        self.wq = nn.Linear(embed_size, embed_size) #Wq
        self.wk = nn.Linear(embed_size, embed_size) #Wk
        self.wv = nn.Linear(embed_size, embed_size) #Wv
        self.wo = nn.Linear(embed_size, embed_size) #Wo
        
        self.dropout = nn.Dropout(p=dropout)
    
    # @staticmethod
    def attention(self, query, key, value, mask, dropout = None):
        dk = query.shape[-1]
        
        #(Batch, h, seq_len, embed_size) * (Batch, h, embed_size, seq_len) --> (Batch, h, seq_len, seq_len)
        attention = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk))
        
        if mask is not None:
            attention.masked_fill_(mask == 0, 1e-9)
            
        attention = F.softmax(attention, dim = -1) #(Batch, h, seq_len, seq_len)
        
        if dropout:
            attention = dropout(attention)
        
        return torch.matmul(attention, value), attention
    
    def forward(self, q, k, v, mask):
        query = self.wq(q)  #(Batch, seq_len, embed_size)
        key = self.wk(k)    #(Batch, seq_len, embed_size)
        value = self.wv(v) #(Batch, seq_len, embed_size)
        
        # pdb.set_trace()
        # batch_size, seq_len, _ = q.size()
        
        #(Batch, seq_len, embed_size) --> (Batch, seq_len, nheads, dk) --> (Batch, nheads, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.nheads, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.nheads, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.nheads, self.dk).transpose(1, 2)
        # pdb.set_trace()
        # x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x, self.attention_scores = self.attention(query, key, value, mask, self.dropout)
        
        #(Batch, h, seq_len, dk) --> (Batch, seq_len, h, dk) --> (Batch, seq_len, embed_size)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.nheads * self.dk)
        
        return self.wo(x)


class EncoderBlock(nn.Module):
    """
    ENCODER BLOCK MODULE
    
    This module represents a single block in the encoder part of the Transformer architecture. It consists
    of a multi-head attention layer, followed by a feed-forward network and layer normalization. The encoder
    block is responsible for encoding the input sequence into a high-level representation.
    """
    def __init__(self, embed_size, nheads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, nheads, dropout = 0.1)
        self.fc = FeedForward(embed_size, d_ff, dropout = 0.1)
        self.layernorm = LayerNorm()
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x, src_mask):
        # pdb.set_trace()
        attn_x = self.self_attention(x, x, x, src_mask)
        x = x + self.dropout(attn_x)
        x = self.layernorm(x)
        
        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.layernorm(x)
        
        return x
    

class Encoder(nn.Module):
    """
    ENCODER MODULE
    
    This module combines multiple encoder blocks into a stack, forming the complete encoder part of the
    Transformer architecture. It takes the input sequence and processes it through the stack of encoder
    blocks to produce the encoded representation.
    """
    def __init__(self, nlayers, embed_size, nheads, d_ff):
        super().__init__()

        encoder_blocks = []
        for i in range(nlayers):
            encoder_blocks.append(EncoderBlock(embed_size, nheads, d_ff))
            
        self.encoder = nn.ModuleList(encoder_blocks)

        
    def forward(self, x, encoder_mask):
        
        for encoder_block in self.encoder:
            x = encoder_block(x, encoder_mask)
        return x #self.encoder(x, encoder_mask)


class DecoderBlock(nn.Module):
    """
    DECODER BLOCK MODULE
    
    This module represents a single block in the decoder part of the Transformer architecture. It consists
    of two multi-head attention layers (one for self-attention and one for cross-attention with the encoder
    output), followed by a feed-forward network and layer normalization. The decoder block is responsible
    for generating the output sequence based on the encoded input and previously generated tokens.
    """
    def __init__(self, embed_size, nheads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, nheads, dropout = 0.5)
        self.cross_attention = MultiHeadAttention(embed_size, nheads, dropout = 0.5)
        self.fc = FeedForward(embed_size, d_ff, dropout = 0.5)
        self.layernorm = LayerNorm()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x, encoder_out, src_mask, tgt_mask):
        selfattn_x = self.self_attention(x, x, x, tgt_mask)
        x = x + self.dropout(selfattn_x)
        x = self.layernorm(x)
        
        crossattn_x = self.cross_attention(x, encoder_out, encoder_out, src_mask)
        x = x + self.dropout(crossattn_x)
        x = self.layernorm(x)
        
        fc_out = self.fc(x)
        x = x + self.dropout(fc_out)
        x = self.layernorm(x)
        
        return x
    

class Decoder(nn.Module):
    """
    DECODER MODULE
    
    This module combines multiple decoder blocks into a stack, forming the complete decoder part of the
    Transformer architecture. It takes the encoded input from the encoder and the target sequence, and
    processes them through the stack of decoder blocks to generate the output sequence.
    """
    def __init__(self, nlayers, embed_size, nheads, d_ff):
        super().__init__()

        decoder_blocks = []
        for i in range(nlayers):
            decoder_blocks.append(DecoderBlock(embed_size, nheads, d_ff))
            
        self.decoder = nn.ModuleList(decoder_blocks)
    def forward(self, decoder_input, encoder_out, encoder_mask, decoder_mask):
        
        for decoder_block in self.decoder:
            x = decoder_block(decoder_input, encoder_out, encoder_mask, decoder_mask)
        return x #self.decoder(decoder_input, encoder_out, encoder_mask, decoder_mask)

class OutputLayer(nn.Module):
    """
    OUTPUT PROJECTION LAYER MODULE
    
    This module is responsible for taking the output of the decoder and projecting it into the target
    vocabulary space. It applies a linear transformation followed by a log-softmax activation to produce
    the final output logits, which can be used for tasks like language generation or machine translation.
    """
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(embed_size, vocab_size)
    
    def forward(self, x):
        
        return F.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    """
    TRANSFORMER MODEL
    
    This class represents the complete Transformer model, combining the encoder, decoder, input embeddings,
    positional encodings, and output layer. It takes the source and target sequences as input and produces
    the output logits, which can be used for various sequence-to-sequence tasks.
    """
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        embed_size = 512,
        n_encoder = 6,
        n_decoder = 6,
        nheads = 8,
        dropout = 0.1,
        d_ff = 2048
        ):
        super().__init__()
        
        self.src_embed = InputEmbeddings(embed_size, src_vocab_size)
        self.tgt_embed = InputEmbeddings(embed_size, tgt_vocab_size)
        
        self.pos_enc = PositionalEncoding(embed_size, max(src_seq_len, tgt_seq_len), dropout)
        
        self.encoder = Encoder(n_encoder, embed_size, nheads, d_ff)
        
        self.decoder = Decoder(n_decoder, embed_size, nheads, d_ff)
        
        self.outputLayer = OutputLayer(embed_size, tgt_vocab_size)
        
    
    def encode(self, encoder_input, encoder_mask):
        x = self.src_embed(encoder_input)
        x = self.pos_enc(x)
        return self.encoder(x, encoder_mask)
    
    def decode(self, decoder_input, encoder_out, encoder_mask, decoder_mask):
        y = self.tgt_embed(decoder_input)
        y = self.pos_enc(y)
        return self.decoder(y, encoder_out, encoder_mask, decoder_mask)
    
    def project(self, decoder_out):
        return self.outputLayer(decoder_out)
        
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        
        x = self.src_embed(encoder_input)
        x = self.pos_enc(x)
        
        y = self.tgt_embed(decoder_input)
        y = self.pos_enc(y)
        
        encoder_out = self.encoder(x, encoder_mask)
        
        decoder_out = self.decoder(y, encoder_out, encoder_mask, decoder_mask)
        
        return self.outputLayer(decoder_out)
        

def build_transformer(src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        embed_size = 512,
        n_encoder = 6,
        n_decoder = 6,
        nheads = 8,
        dropout = 0.1,
        d_ff = 2048):

    transformer = Transformer(src_vocab_size,
        tgt_vocab_size,
        src_seq_len,
        tgt_seq_len,
        embed_size = 512,
        n_encoder = 6,
        n_decoder = 6,
        nheads = 8,
        dropout = 0.1,
        d_ff = 2048)
    
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    
    
    return transformer
