import torch
import torch.nn.functional as F
from torch import nn

from models.stacked.sentence_encoder import BertSentenceEncoder

import math

class StackedBert(nn.Module):

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        #input_channels = 1
        #ks = 3

        self.sentence_encoder = BertSentenceEncoder.from_pretrained(
            args.pretrained_model_path, num_labels=args.num_labels)
        
        """
        self.conv1 = nn.Conv2d(input_channels,
                               args.output_channel,
                               (3, self.sentence_encoder.config.hidden_size),
                               padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channels,
                               args.output_channel,
                               (4, self.sentence_encoder.config.hidden_size),
                               padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channels,
                               args.output_channel,
                               (5, self.sentence_encoder.config.hidden_size),
                               padding=(4, 0))

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(ks * args.output_channel, args.num_labels)
        """

        # FT
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(args.ninp, args.dropout)
        encoder_layers = TransformerEncoderLayer(args.ninp, args.nhead, args.nhid, args.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, args.nlayers)
        self.encoder = nn.Embedding(args.ntoken, args.ninp)
        self.ninp = args.ninp
        self.decoder = nn.Linear(args.ninp, args.ntoken)

        self.init_weights()


    # FT
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    # FT
    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_ids, segment_ids=None, input_mask=None):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        segment_ids = segment_ids.permute(1, 0, 2)
        input_mask = input_mask.permute(1, 0, 2)

        x_encoded = []
        for i0 in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i0], input_mask[i0], segment_ids[i0]))

        x = torch.stack(x_encoded)  # (sentences, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)

        """
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]

        if self.args.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch_size, output_channels) * ks
            x = torch.cat(x, 1)  # (batch_size, output_channels * ks)
            x = x.view(-1, self.filter_widths * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences) * ks
            x = torch.cat(x, 1)  # (batch_size, channel_output * ks)

        x = self.dropout(x)
        logits = self.fc1(x)  # (batch_size, num_labels)

        #return logits, x
        """
        
        has_mask = False # TODO: implement with mask
        # transforemr
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        
        """
        generating mask
        """
        src = x
        mask = self._generate_square_subsequent_mask(len(src)) #.to(device)
        self.src_mask = mask
        #

        #src = self.encoder(src) * math.sqrt(self.ninp)
        
        src = self.pos_encoder(src)
        
        DOCLEVEL = True
        if DOCLEVEL:
            # take the last sentence's representation as the representative
            output = self.transformer_encoder(src, self.src_mask)[:,0,:]
        else:
            # sentence level, all sentence vectors are taken
            output = self.transformer_encoder(src, self.src_mask)

        #output = self.decoder(output)
        #return F.log_softmax(output, dim=-1)
        
        # why not time distributed here?
        logits = self.decoder(output) #.squeeze()
        #print("input ids:", input_ids.size())
        #print("logits:", logits.size())
        #print("output:", output.size())
        return logits, output

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


