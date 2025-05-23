from sklearn.feature_extraction.text import TfidfVectorizer
import torch
def TF_IDF(data, args):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data)
    tfidf_embedding = torch.tensor(tfidf_matrix.toarray())
    torch.save(tfidf_embedding, f'{args.output_dir}/{args.dataset}_tfidf.pt')
