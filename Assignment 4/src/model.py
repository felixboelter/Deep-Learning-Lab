import torch
from torch import nn
import torchtext
import math

# The positional encoding is a sinusoid that is added to the embedding vector. The sinusoid has a
# different frequency for each dimension of the embedding vector
class PositionalEncoding(nn.Module):
  #Positional Encoding taken from the slides.
  def __init__(self, d_model, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.max_len = max_len

    pe = torch.zeros(max_len, d_model)
    position = torch.arange(
        0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                          * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)
  
  def forward(self, x):
    assert x.size(0) < self.max_len, (
        f"Too long sequence: increase 'max_len'"
    )
    x = x + self.pe[:x.size(0), :]
    return x

class TransformerModel(nn.Module):
  def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                dim_feedforward: int = 2048,src_tokens: int = None,tgt_tokens: int = None,
                device : torch.cuda.device = None, target_field : torchtext.data.Field = None) -> None:
    """
    The TransformerModel class is a subclass of nn.Module. It takes in a number of arguments, and then
    initializes the superclass, and then initializes the transformer, positional encoder, source and
    target embeddings, and the fully connected layer
    
    :param d_model: The number of expected features in the encoder/decoder inputs (default=512),
    defaults to 512
    :type d_model: int (optional)
    :param nhead: the number of heads in the multiheadattention models (default=8), defaults to 8
    :type nhead: int (optional)
    :param num_encoder_layers: The number of encoder layers in the transformer, defaults to 6
    :type num_encoder_layers: int (optional)
    :param num_decoder_layers: The number of decoder layers, defaults to 6
    :type num_decoder_layers: int (optional)
    :param dim_feedforward: The dimension of the feedforward network model inside the transformer,
    defaults to 2048
    :type dim_feedforward: int (optional)
    :param src_tokens: The number of tokens in the source vocabulary
    :type src_tokens: int
    :param tgt_tokens: The number of tokens in the target language
    :type tgt_tokens: int
    :param device: The device to run the model on
    :type device: torch.cuda.device
    :param target_field: The field that contains the target data
    :type target_field: Field
    """
    
    super(TransformerModel, self).__init__()
    self.d_model = d_model
    self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
    self.pos_encoder = PositionalEncoding(d_model)
    self.src_embedding = nn.Embedding(src_tokens, d_model)
    self.tgt_embedding = nn.Embedding(tgt_tokens,d_model)
    self.fc = nn.Linear(d_model,tgt_tokens)
    self.device = device
    self.target_field = target_field
    
  def generate_subsequent_mask(self, sz):
    #Taken from nn.Transformer
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
  def generate_padded_mask(self, pad):
    padded_mask = (pad == self.target_field.vocab.stoi[self.target_field.pad_token])
    return padded_mask
  def generate_masks_and_encoding(self, input, src_embedding = True):
    """
    This function generates the masks and positional encodings for the input
    
    :param input: The input sequence
    :param src_embedding: This is a boolean value that tells the function whether to use the source
    embedding or the target embedding, defaults to True (optional)
    :return: The input_key_padding_mask is a mask that is used to mask out the padded tokens in the
    input.
    The input_mask is a mask that is used to mask out the tokens that are after the current token.
    The input_pos_encoder is the positional encoding of the input.
    """
    input_key_padding_mask = self.generate_padded_mask(input)
    input_mask = self.generate_subsequent_mask(input.size(1))
    if src_embedding == True:
      input_pos_encoder = self.pos_encoder(self.src_embedding(input).transpose(1,0))
      memory_key_padding_mask = input_key_padding_mask.clone()
      return input_key_padding_mask.to(self.device),memory_key_padding_mask.to(self.device),input_mask.to(self.device),input_pos_encoder.to(self.device)
    else:
      input_pos_encoder = self.pos_encoder(self.tgt_embedding(input).transpose(1,0))
      return input_key_padding_mask.to(self.device),input_mask.to(self.device),input_pos_encoder.to(self.device)

  def forward(self, src ,tgt):
    """
    The function takes in the source and target sentences, generates masks and encodings for both, and
    then passes them to the transformer model
    
    :param src: The source sequence
    :param tgt: The target sequence
    :return: The output of the last layer of the transformer model.
    """
    src_key_padding_mask,memory_key_padding_mask,_,src_pos_encoder = self.generate_masks_and_encoding(src,src_embedding=True)
    tgt_key_padding_mask,tgt_mask,tgt_pos_encoder = self.generate_masks_and_encoding(tgt,src_embedding=False)
    out = self.transformer(src_pos_encoder, tgt_pos_encoder, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
    out = self.fc(out.transpose(1,0))
    return out