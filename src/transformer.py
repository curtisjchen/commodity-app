from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch

class CommodityClassifier(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        # 1. Flexible Backbone
        self.transformer = AutoModel.from_pretrained(hparams['backbone_name'])
        config = AutoConfig.from_pretrained(hparams['backbone_name'])
        text_dim = self.transformer.config.hidden_size # 768 for BERT, 1024 for Large, etc.
        
        # 2. Flexible Categorical Embedding (e.g., Department)
        self.cat_embed = nn.Embedding(num_embeddings=50, embedding_dim=hparams['cat_dim'])
        
        # 3. Dynamic MLP Head
        all_layers = []
        input_dim = text_dim + hparams['cat_dim'] # Concatenated size
        
        for h_size in hparams['hidden_layers']:
            all_layers.append(nn.Linear(input_dim, h_size))
            all_layers.append(nn.ReLU())
            all_layers.append(nn.Dropout(0.3))
            input_dim = h_size # Next layer input is current layer output
            
        all_layers.append(nn.Linear(input_dim, hparams['num_labels']))
        self.classifier = nn.Sequential(*all_layers)

    def forward(self, input_ids, attention_mask, cat_idx):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        cat_feat = self.cat_embed(cat_idx)
        combined = torch.cat((text_feat, cat_feat), dim=1)
        
        logits = self.classifier(combined)
        return logits