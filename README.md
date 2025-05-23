# Cequel

This code is used for running experiments for the paper.

1. **Data**: Raw data has been included in the `data` folder.

2. **Embedding**: You can check the `baselines` folder and generate embeddings of all encoders used in the paper, like:

   ```python
   python generate_embedding.py --model tfidf --dataset tweet --output_dir new_embeddings
   ```

   Or you can download the `instructor` and `sentencebert` embeddings from this link https://www.dropbox.com/scl/fo/vzlvs2quhg90l5v9821bk/AJndRy9yAQMjZyou0THB7nE?rlkey=kswrq8aghbv0b2jz25vrafrxj&st=lem6dl1t&dl=0

3. **Baselines**: When embeddings are ready, run the `embedding_clustering.py` in the `baselines` folder:

   ```
   python embedding_clustering.py
   ```

4. **Requirements**: Install all dependencies.

   ```
   conda create --name Cequel python=3.12.3
   pip install scikit-learn
   pip install sentence_transformers
   pip install datasets
   pip install InstructorEmbedding
   pip install nltk
   pip install gensim
   pip install ortools
   pip install jsonlines
   pip install openai
   pip install metric-learn
   pip install sentence-transformers==2.2.2
   pip install huggingface_hub==0.25.0
   pip install openai==0.28
   ```

5. **Main experiments**: Before running the `main.py` to get results, please replace your api key in our **Edge LLM** and **TriangleLLM** in `llm_clustering\active_semi_supervised_clustering\active_semi_clustering\active\pairwise_constraints\gpt3_pc_oracle_edge.py` and `llm_clustering\active_semi_supervised_clustering\active_semi_clustering\active\pairwise_constraints\gpt3_pc_oracle_triangle.py`, and check the parameters in it.

   For example:

   ```
   python main.py --corpus_name BBC_News --selection triangle --encoder_name instructor --clustering WeightedPCKMeans --mode max-sum --weight_method log_degree_ratio --eigen 0.1
   ```

6. **Results**: Check the prompts, responses, and final results.

