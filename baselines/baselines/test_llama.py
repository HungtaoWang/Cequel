from transformers import LlamaTokenizer, LlamaModel
import torch


def llama(data, args):
    llama_path = "workspace/co-training-pp/models/Llama-2-7b-hf"
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    model = LlamaModel.from_pretrained(llama_path).to(args.device)
    node_predictions = None
    for text in data:
        node_input = tokenizer(text, return_tensors='pt').to(args.device)
        with torch.no_grad():
            node_output = model(**node_input)
        if node_predictions is None:
            node_predictions = node_output.last_hidden_state.mean(dim=1)
        else:
            node_predictions = torch.cat((node_predictions, node_output.last_hidden_state.mean(dim=1)), dim=0)
    torch.save(node_predictions, f'{args.output_dir}/{args.dataset}_llama.pt')
