import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
import ml_collections
import math
import types

class AdapterBlock(nn.Module):
    def __init__(self, input_dim, adapter_hidden_dim, dropout):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, adapter_hidden_dim)
        self.linear2 = nn.Linear(adapter_hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        # initialize weights to a small constant
        for module in [self.linear1, self.linear2]:
            nn.init.normal_(module.weight, 0, .01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # down-project
        u = F.relu(self.linear1(self.dropout(x)))
        # up-project
        u = self.linear2(u)
        # skip connection
        u = x + u
        return u

class LoRABlock(nn.Module):
    def __init__(self, in_features, out_features, alpha=8, r=8):
        super(LoRABlock, self).__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(r).float())
        self.A = nn.Parameter(torch.randn(in_features, r) * std_dev)
        self.B = nn.Parameter(torch.zeros(r, out_features))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

class RobertaPEFT(nn.Module):
    def __init__(self, num_classes=2, args=None):
        super().__init__()
        self.args = args
        self.config = RobertaConfig.from_pretrained(args.model_name)
        self.roberta = RobertaModel.from_pretrained(args.model_name, add_pooling_layer=False)
        if args.peftmode != 'full-ft':
            for param in self.roberta.parameters():
                param.requires_grad = False

        # PEFT configurations
        self.peftmode = args.peftmode
        self.lora_cfg = None
        self.adapter_cfg = None

        if args.peftmode == 'lora':
            self.lora_cfg = ml_collections.ConfigDict()
            self.lora_cfg.alpha = args.alpha
            self.lora_cfg.r = args.r
            self._add_lora_layers()
        elif args.peftmode == 'adapter':
            self.adapter_cfg = ml_collections.ConfigDict()
            self.adapter_cfg.dim = args.adapter_dim
            self.adapter_cfg.dropout = 0
            self._add_adapter_layers()
        elif args.peftmode == 'bitfit':
            self._setup_bitfit()

        self.head = nn.Linear(self.config.hidden_size, num_classes)

    def _add_lora_layers(self):
        for i, layer in enumerate(self.roberta.encoder.layer):
            # Add LoRA to query and value projections
            layer.attention.self.query.lora = LoRABlock(
                self.config.hidden_size,
                self.config.hidden_size,
                self.lora_cfg.alpha,
                self.lora_cfg.r
            )
            layer.attention.self.value.lora = LoRABlock(
                self.config.hidden_size,
                self.config.hidden_size,
                self.lora_cfg.alpha,
                self.lora_cfg.r
            )

            # Store original forward method
            layer.attention.self.query.org_forward = layer.attention.self.query.forward
            layer.attention.self.value.org_forward = layer.attention.self.value.forward


            def new_forward(self, hidden_states, *args, **kwargs):
                return self.org_forward(hidden_states, *args, **kwargs) + self.lora(hidden_states)

            # Replace the forward method
            layer.attention.self.query.forward = types.MethodType(new_forward, layer.attention.self.query)
            layer.attention.self.value.forward = types.MethodType(new_forward, layer.attention.self.value)

    def _add_adapter_layers(self):
        for i, layer in enumerate(self.roberta.encoder.layer):
            layer.attention.output.dense.adapter = AdapterBlock(
                self.config.hidden_size,
                self.adapter_cfg.dim,
                dropout=self.adapter_cfg.dropout
            )

            layer.output.dense.adapter = AdapterBlock(
                self.config.hidden_size,
                self.adapter_cfg.dim,
                dropout=self.adapter_cfg.dropout
            )
            # Store original forward method
            layer.attention.output.dense.org_forward = layer.attention.output.dense.forward
            layer.output.dense.org_forward = layer.output.dense.forward
            def new_forward(self, hidden_states, *args, **kwargs):
                return self.adapter(self.org_forward(hidden_states, *args, **kwargs))

            # Replace the forward method
            layer.attention.output.dense.forward = types.MethodType(new_forward, layer.attention.output.dense)
            layer.output.dense.forward = types.MethodType(new_forward, layer.output.dense)

    def _setup_bitfit(self):
        # Freeze all parameters except bias terms
        for name, param in self.roberta.named_parameters():
            if 'bias' in name:
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None):
        if isinstance(input_ids, tuple):
            input_ids, attention_mask = input_ids
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get [CLS] token output
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.head(pooled_output)

        return {'logits': logits, 'cls_token': pooled_output}