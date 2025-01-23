# QEEC

This code is used for running experiments for the paper.

1. Download the dataset file from https://www.dropbox.com/scl/fo/n2qmmg7wr2gqjzp42xy50/ANaZR9sswhoee_iVwnkY5bQ?rlkey=xvza2s3pndprpbhkpcexxd7o6&st=84bi6ew2&dl=0.

2. Install all dependencies.

   ```
   conda create --name QEEC python=3.12.3
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

   

3. Run the `main.py` to get results, and check the parameters in it.

4. Check the results in the terminal.