# from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbModel(nn.Module):
    def __init__(self, num_classes, pretrained_model_dir):
        super(EmbModel, self).__init__()
        print("Initializing Model Embedding...")
        # self.pretrained = AutoModel.from_pretrained(pretrained_model_dir)
        self.pretrained = torch.load(pretrained_model_dir)
        self.fc = nn.Linear(384, num_classes)

    def forward(self, sentences):
        mapping = {
            'input_ids' : sentences[:,:,0].to(torch.int64),
            'token_type_ids':sentences[:,:,1].to(torch.int64),
            'attention_mask':sentences[:,:,2].to(torch.int64)
        }
        x = self.pretrained(**mapping)
        x = self.mean_pooling(x, mapping['attention_mask'])
        x = F.normalize(x, p=2, dim=1)
        return self.fc(x)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def pipeline(self, tokenizer, sentences):
        token = tokenizer(sentences, padding='max_length', max_length=30, return_tensors='pt')
        token_tensor = torch.dstack([
                token['input_ids'], 
                token['token_type_ids'], 
                token['attention_mask']
            ])
        with torch.no_grad():
            out = self(token_tensor)
        return torch.argmax(out.detach())

# new_model = EmbModel(431, 384)
