import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from lib.qa_helper import *
    

class LangModule(nn.Module):
    def __init__(self, use_bidir=False, num_layers=1,
        emb_size=300, hidden_size=768, pdrop=0.1, 
        use_bert=False, bert_model_name=None, freeze_bert=False, 
        finetune_bert_last_layer=False, finetune_bert_full=False):
        super().__init__() 

        self.use_bidir = use_bidir
        self.num_layers = num_layers       
        self.use_bert = use_bert        
        self.bert_model_name = bert_model_name

        if self.use_bert:
            # to use transformers, install it first: pip install transformers
            from transformers import AutoModel 
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
            # assert not (freeze_bert and finetune_bert_last_layer)
            if freeze_bert:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            elif finetune_bert_last_layer:
                for param in self.bert_model.parameters():
                    param.requires_grad = False
                if hasattr(self.bert_model, 'encoder'):
                    for param in self.bert_model.encoder.layer[-1].parameters():
                        param.requires_grad = True
                else: # distill-bert
                    for param in self.bert_model.transformer.layer[-1].parameters():
                        param.requires_grad = True    
            elif finetune_bert_full:
                for param in self.bert_model.parameters():
                    param.requires_grad = True
            else:
                raise ValueError('invalid bert finetune mode!')


        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
            bidirectional=use_bidir,
            dropout=0.1 if num_layers > 1 else 0,
        )

        self.word_drop = nn.Dropout(pdrop)

        lang_size = hidden_size * 2 if use_bidir else hidden_size


    def make_mask(self, feature):
        """
        return a mask that is True for zero values and False for other values.
        """
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0) #.unsqueeze(-1) #.unsqueeze(2)        


    def forward(self, data_dict):
        """
        encode the input descriptions
        """
        
        if self.use_bert:
            s_embs = (self.bert_model(**data_dict["s_feat"])).last_hidden_state
            q_embs = (self.bert_model(**data_dict["q_feat"])).last_hidden_state  
            data_dict["s_out"] = s_embs
            data_dict["q_out"] = q_embs
            data_dict["s_mask"] = ~data_dict["s_feat"]["attention_mask"].bool()
            data_dict["q_mask"] = ~data_dict["q_feat"]["attention_mask"].bool()

        else:
            s_embs = data_dict["s_feat"]  # torch.Size([32, 100, 300]) 300 is glove dim, 100 is max len
            q_embs = data_dict["q_feat"]  # torch.Size([32, 100, 300])

            # dropout word embeddings
            s_embs = self.word_drop(s_embs)
            q_embs = self.word_drop(q_embs)

            s_feat = pack_padded_sequence(s_embs, data_dict["s_len"].cpu(), batch_first=True, enforce_sorted=False)
            q_feat = pack_padded_sequence(q_embs, data_dict["q_len"].cpu(), batch_first=True, enforce_sorted=False)

            # encode description
            packed_s, (_, _) = self.lstm(s_feat)
            packed_q, (_, _) = self.lstm(q_feat)
            
            s_output, _ = pad_packed_sequence(packed_s, batch_first=True)
            q_output, _ = pad_packed_sequence(packed_q, batch_first=True)
            
            data_dict["s_out"] = s_output # torch.Size([32, X(48), 256])
            data_dict["q_out"] = q_output # torch.Size([32, Y(16), 256])
            data_dict["s_mask"] = self.make_mask(s_output) # torch.Size([32, X(48)])
            data_dict["q_mask"] = self.make_mask(q_output) # torch.Size([32, Y(16)])

            # import pdb; pdb.set_trace()

        return data_dict
