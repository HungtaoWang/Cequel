from sentence_transformers import SentenceTransformer
import torch
def test_instruct(data, args):
    model = SentenceTransformer("hkunlp/instructor-large").to(args.device)
    embeddings = torch.tensor(model.encode(data))
    torch.save(embeddings, f'{args.output_dir}/{args.dataset}_instructor-large.pt')
