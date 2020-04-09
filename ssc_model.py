import logging
from typing import Dict

#from torch.nn import LayerNorm
import torch
from torch.nn import Linear
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, TimeDistributed, Seq2SeqEncoder, LayerNorm
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import F1Measure, CategoricalAccuracy
from allennlp.modules.conditional_random_field import ConditionalRandomField
import numpy as np

#import torch
import torch.nn.functional as F
from torch import nn

from torch import nn
from transformers import BertPreTrainedModel, BertModel


import math

from allennlp.nn.util import get_text_field_mask


logger = logging.getLogger(__name__)

@Model.register("SeqClassificationModel")
class SeqClassificationModel(Model):
    """
    Question answering model where answers are sentences
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 use_sep: bool = True,
                 concat: bool = False,
                 with_crf: bool = False,
                 self_attn: Seq2SeqEncoder = None,
                 bert_dropout: float = 0.1,
                 sci_sum: bool = False,
                 additional_feature_size: int = 0,
                 lda_dist_size: int = 8,
                 weighted = False #True
                 ) -> None:
        super(SeqClassificationModel, self).__init__(vocab)

        #self.text_field_embedder = text_field_embedder
        self.vocab = vocab
        self.use_sep = use_sep
        self.with_crf = with_crf
        self.sci_sum = sci_sum
        self.self_attn = self_attn
        self.additional_feature_size = additional_feature_size
        
        """
        """
        pretrained_model_path = "/scratch/mihalcea_root/mihalcea/dojmin/stacked/hedwig-data/models/bert_pretrained/bert-base-uncased"
        self.num_labels = self.vocab.get_vocab_size(namespace='labels')
        ninp = 768 # bert base-uncased dim
        nhid = 200
        nlayers = 2
        nhead = 2
        ntoken = self.num_labels
        self.sentence_encoder = BertSentenceEncoder.from_pretrained(
            pretrained_model_path, num_labels=self.num_labels)

        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, bert_dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, bert_dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(token, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()
        
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.labels_are_scores = False
        self.num_labels = self.vocab.get_vocab_size(namespace='labels')
            # define accuracy metrics
        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}

            # define F1 metrics per label
        for label_index in range(self.num_labels):
            label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
            self.label_f1_metrics[label_name] = F1Measure(label_index)


        return


    def forward(self,  # type: ignore
                sentences: torch.LongTensor,
                labels: torch.IntTensor = None,
                confidences: torch.Tensor = None,
                additional_features: torch.Tensor = None,
                lda_dist: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        TODO: add description

        Returns
        -------
        An output dictionary consisting of:
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        #print("sentences", sentences)
        #print("labels", labels)

        #print(sentences.size())
        #embedded_sentences = self.text_field_embedder(sentences)
        #print("embedded", embedded_sentences)
        #print(embedded_sentences.size())
        
        input_ids = sentences["bert"]
        segment_ids = sentences["bert-type-ids"]
        offsets = sentences["bert-offsets"]
        input_mask = get_text_field_mask({"bert": input_ids, "bert-type-ids": segment_ids, "bert-offsets":offsets },1) # sentences["mask"] 
        
       
        #print(input_mask)
        #print(input_ids.size())
        #print(segment_ids.size())
        #print(input_mask.size())
 
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        segment_ids = segment_ids.permute(1, 0, 2)
        input_mask = input_mask.permute(1, 0, 2)
        
       
        x_encoded = []
        for i0 in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i0], input_mask[i0], segment_ids[i0]))

        x = torch.stack(x_encoded)  # (sentences, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)

        batch_size, num_sentences,  _ =  x.size()

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
        device = src.device #TODO: confirm
        mask = self._generate_square_subsequent_mask(len(src)).to(device) #TODO: confirm
        self.src_mask = mask
        #

        #src = self.encoder(src) * math.sqrt(self.ninp)
        
        src = self.pos_encoder(src)
        
        DOCLEVEL = False
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
        
        
        label_logits = logits
        label_probs = torch.nn.functional.softmax(label_logits, dim=-1)
            
        #print(label_logits.size())

        # Create output dictionary for the trainer
        # Compute loss and epoch metrics
        output_dict = {"action_probs": label_probs}
    
        if labels is not None:
            # Compute cross entropy loss
            flattened_logits = label_logits.view((batch_size * num_sentences), self.num_labels)
            flattened_gold = labels.contiguous().view(-1)

            if not self.with_crf:
                
                
                # encoutners problems when we have single output
                
                if len(flattened_logits.squeeze().shape) == 1:
                    label_loss = self.loss(flattened_logits.view(1,-1), flattened_gold)
                else:
                #if True:
                    label_loss = self.loss(flattened_logits.squeeze(), flattened_gold)

                if confidences is not None:
                    label_loss = label_loss * confidences.type_as(label_loss).view(-1)
                label_loss = label_loss.mean()
                flattened_probs = torch.softmax(flattened_logits, dim=-1)
            

            if not self.labels_are_scores:
                
                
                if len(flattened_probs.float().contiguous().squeeze().shape) == 1:
                    evaluation_mask = (flattened_gold != -1)
                    self.label_accuracy(flattened_probs.float().contiguous().view(1,-1), flattened_gold, mask=evaluation_mask)
                else:
                #if True:
                    evaluation_mask = (flattened_gold != -1)
                    self.label_accuracy(flattened_probs.float().contiguous(), flattened_gold.squeeze(-1), mask=evaluation_mask)
                
                # compute F1 per label
                for label_index in range(self.num_labels):
                    label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                    metric = self.label_f1_metrics[label_name]
                    metric(flattened_probs, flattened_gold, mask=evaluation_mask)
        
        if labels is not None:
            output_dict["loss"] = label_loss
        output_dict['action_logits'] = label_logits

        return output_dict



    def get_metrics(self, reset: bool = False):
        metric_dict = {}

        if not self.labels_are_scores:
            type_accuracy = self.label_accuracy.get_metric(reset)
            metric_dict['acc'] = type_accuracy

            average_F1 = 0.0
            for name, metric in self.label_f1_metrics.items():
                metric_val = metric.get_metric(reset)
                metric_dict[name + 'F'] = metric_val[2]
                average_F1 += metric_val[2]

            average_F1 /= len(self.label_f1_metrics.items())
            metric_dict['avgF'] = average_F1

        return metric_dict


    # FT
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    # FT
    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)




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

class BertSentenceEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, num_labels=config.num_labels)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pooled_output = self.bert(input_ids, attention_mask, token_type_ids)[1]
        pooled_output = self.dropout(pooled_output)
        return pooled_output
