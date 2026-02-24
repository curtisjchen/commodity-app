import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class CommodityClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load the backbone (DistilBERT, RoBERTa, etc.)
        self.transformer = AutoModel.from_pretrained(config['model_name'])
        
        # Calculate the total input size for the classifier head
        # cls_token_dim + categorical_dim
        input_dim = self.transformer.config.hidden_size + config['cat_dim']
        
        # Build a flexible hidden layer stack
        layers = []
        curr_dim = input_dim
        for h_dim in config['hidden_layers']:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config['dropout']))
            curr_dim = h_dim
        
        # Final output layer
        layers.append(nn.Linear(curr_dim, config['num_classes']))
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, cat_features):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Concatenate with categorical features
        combined = torch.cat((cls_output, cat_features), dim=1)
        return self.classifier(combined)