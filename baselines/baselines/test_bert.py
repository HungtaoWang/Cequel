from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel
import torch
def bert_uncased(data, args, bert_name = 'bert'):
    if bert_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(args.device)
    elif bert_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base').to(args.device)
    elif bert_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(args.device)
    elif bert_name == 'distilroberta':
        tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
        model = RobertaModel.from_pretrained('distilroberta-base').to(args.device)
    node_predictions = None
    for text in data:
        node_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(args.device)
        with torch.no_grad():
            node_output = model(**node_input)
        if node_predictions is None:
            if bert_name == 'distilbert':
                node_predictions = node_output.last_hidden_state[:, 0, :]
            else:
                node_predictions = node_output.pooler_output
        else:
            if bert_name == 'distilbert' :
                node_predictions = torch.cat((node_predictions, node_output.last_hidden_state[:, 0, :]), dim=0)
            else:
                node_predictions = torch.cat((node_predictions, node_output.pooler_output), dim=0)
    torch.save(node_predictions, f'{args.output_dir}/{args.dataset}_{bert_name}.pt')

def sentence_bert(data, args, bert_name = 'sentence_bert'):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2").to(args.device)
    node_predictions = None
    for text in data:
        node_input = model.encode(text, convert_to_tensor=True).to(args.device).unsqueeze(0)
       
        if node_predictions is None:
            node_predictions = node_input
        else:
            node_predictions = torch.cat((node_predictions, node_input), dim=0)
    torch.save(node_predictions, f'{args.output_dir}/{args.dataset}_{bert_name}.pt')



