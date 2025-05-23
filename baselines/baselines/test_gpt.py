from langchain_openai import OpenAIEmbeddings
import torch
# replace the following with your own OpenAI API key and base URL
base_url = ''
api_key = ''
model_text = OpenAIEmbeddings(
    model='text-embedding-ada-002',
    base_url=base_url,
    api_key=api_key
    )

def embed_text(data, args):
    res = model_text.embed_documents(data)
    torch.save(res, f'{args.output_dir}/{args.dataset}_openai_text-embedding-ada-002.pt')