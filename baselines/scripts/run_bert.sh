for model in 'bert' 'roberta' 'distilbert' 'distilroberta' 'sentence_bert'
    do
    for dataset in 'bbc' 'tweet' 'bank77' 'clinic' 'massive_scenario' 'reddit'
        do
            python generate_embedding.py --model $model --dataset $dataset
        done
    done