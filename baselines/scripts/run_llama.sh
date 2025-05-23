for dataset in 'bbc' 'tweet' 'bank77' 'clinic' 'massive_scenario' 'reddit'
    do
        python generate_embedding.py --model 'llama' --dataset $dataset
    done