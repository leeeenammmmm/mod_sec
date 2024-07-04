import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from torch.autograd import Function
import __main__

config = {
    "num_labels": 6,
    "hidden_dropout_prob": 0.3,
    "hidden_size": 768,
    "max_length": 512,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_dict ={0: '000 - Normal',
             1: '126 - Path Traversal',
             2: '242 - Code Injection',
             3: '274 - HTTP Verb Tampering',
             4: '66 - SQL Injection',
             5: '88 - OS Command Injection'}

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
    
class DomainAdaptationModel(nn.Module):
    def __init__(self):
        super(DomainAdaptationModel, self).__init__()
        
        num_labels = config["num_labels"]
        self.bert_model  = AutoModel.from_pretrained('jackaduma/SecBERT')
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.linear = torch.nn.Linear(768, 6)


    def forward(
          self,
          input_ids=None,
          attention_mask=None,
          token_type_ids=None,
          labels=None,
          grl_lambda = 1.0, 
          ):

        outputs = self.bert_model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

#         pooled_output = outputs[1] # For bert-base-uncase
        output_dropout = self.dropout(outputs.pooler_output)


        reversed_pooled_output = GradientReversalFn.apply(output_dropout, grl_lambda)

        request_pred = self.linear(output_dropout)
        domain_pred = self.linear(reversed_pooled_output)

        return request_pred.to(device), domain_pred.to(device)

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')

    def __getitem__(self, index):
        review = self.df.iloc[index]["text"]
        request = self.df.iloc[index]["label"]
        request_dict = {'Injection': 0,
          'Manipulation': 1,
          'Scanning for Vulnerable Software': 2,
          'HTTP abusion': 3,
          'Fake the Source of Data': 4}
        label = request_dict[request]
        encoded_input = self.tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                max_length= config["max_length"],
                truncation=True,
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None



        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]



    def __len__(self):
        return self.df.shape[0]

class UnknownAttackClassificationModel():
    def __init__(self):
        print("loaded model")
        self.model = DomainAdaptationModel()

    # init model
    def loadModelInit(self):
        setattr(__main__, "DomainAdaptationModel", DomainAdaptationModel)
        model_path = "/mnt/e/datn/Transformer-Explainability/bert_models/capec/classifier/SecBert_6labels_balance.pt"
        self.model.load_state_dict(torch.load(model_path)['state_dict'])
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained('jackaduma/SecBERT')

    # Predict
    def predict(self, data):
        encoded_input = self.tokenizer.encode_plus(
            data,
            add_special_tokens=True,
            max_length= config["max_length"],
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        request_pred, _ = self.model(input_ids, attention_mask)
        pred_label = request_pred.max(dim = 1)[1]
        return label_dict[pred_label.item()]