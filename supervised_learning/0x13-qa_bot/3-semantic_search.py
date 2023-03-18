#!/usr/bin/env python3
"""
Semantic search over documents to find the best document to search in for an
answer.

We will use FAII to compare embeddings of the document
with the question asked.
"""
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModel

def cls_pooling(model_output):
    """Cls pooling for classification tasks"""
    return model_output.last_hidden_state[:, 0]

def semantic_search(corpus_path, sentence):
    """Perform Semantic."""
    corpus = load_dataset("text", data_dir=corpus_path, sample_by="document")['train']
    # print(corpus['train'])
    # exit()

    # embedding with sentence transformers
    model_ckpt = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = TFAutoModel.from_pretrained(model_ckpt)

    def get_embedding(sentence_list):
        encoded_input = tokenizer(
            sentence_list, padding=True, truncation=True, return_tensors="tf"
        )
        model_output = model(**encoded_input)
        return cls_pooling(model_output)
    embeddings_corpus = corpus.map(
        lambda x: {"embeddings": get_embedding(x['text']).numpy()[0]}
    )
    question_embedding = get_embedding(sentence).numpy()

    embeddings_corpus.add_faiss_index(column="embeddings")
    _, samples = embeddings_corpus.get_nearest_examples(
        "embeddings", question_embedding, k=1
    )
    
    return samples['text'][0]
