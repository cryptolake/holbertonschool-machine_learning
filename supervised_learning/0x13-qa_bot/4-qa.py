#!/usr/bin/env python3
"""
QA BOT.

QA bot using pretrained bert with local documents
as dictionary.
make sure to do: pip3 install datasets 
"""
import tensorflow as tf
import tensorflow_hub as hub
from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModel, BertTokenizer

model_ckpt = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
SEMANTIC_tok = AutoTokenizer.from_pretrained(model_ckpt)
SEMANTIC_mod = TFAutoModel.from_pretrained(model_ckpt)


def cls_pooling(model_output):
    """Cls pooling for classification tasks"""
    return model_output.last_hidden_state[:, 0]

def get_embedding(sentence_list):
    encoded_input = SEMANTIC_tok(
        sentence_list, padding=True, truncation=True, return_tensors="tf"
    )
    model_output = SEMANTIC_mod(**encoded_input)
    return cls_pooling(model_output)


def create_semantic_model(corpus_path):
    """create model for semantic search"""
    corpus = load_dataset("text", data_dir=corpus_path, sample_by="document")['train']

    # embedding with sentence transformers

    embeddings_corpus = corpus.map(
        lambda x: {"embeddings": get_embedding(x['text']).numpy()[0]}
    )
    embeddings_corpus.add_faiss_index(column="embeddings")
    return embeddings_corpus

def semantic_search(embeddings_corpus, sentence):
    """Perform Semantic."""
    question_embedding = get_embedding(sentence).numpy()

    _, samples = embeddings_corpus.get_nearest_examples(
        "embeddings", question_embedding, k=1
    )
    
    return samples['text'][0]


def question_ref(question, reference):
    """
    Load Bert model and use refrence to answer question.

    question: question string
    reference: reference string containing answer

    Return:
        answer: the question of the question given the reference string
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    tok_que = tokenizer.tokenize(question)
    tok_ref = tokenizer.tokenize(reference)

    tokens = ['[CLS]'] + tok_que + ['[SEP]'] + tok_ref + ['[SEP]']
    word_ids = tokenizer.convert_tokens_to_ids(tokens)
    mask = [1] * len(word_ids)
    # we mask the refrence strings with 0 and question string with 1
    type_ids = [0] * (1 + len(tok_que) + 1) + [1] * (len(tok_ref) + 1)
    word_ids, mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
                                   (word_ids, mask, type_ids))

    outputs = model([word_ids, mask, type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]`
    # is the ignored '[CLS]' token logit

    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    if short_start >= short_end:
        return None
    # print(outputs[0][0][1:][short_start-1], outputs[1][0][1:][short_end-1])

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    # answer2 = tokenizer.convert_tokens_to_string(outputs)
    return answer


def question_answer(corpus_path):
    """
    Main question/answer loop.
    """
    question = ""
    quit_str = ['exit', 'quit', 'goodbye', 'bye']
    embeddings_corpus = create_semantic_model(corpus_path)

    while True:
        question = input("Q: ")
        if question.lower() in quit_str:
            break
        reference = semantic_search(embeddings_corpus, question)
        answer = question_ref(question, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."
        print("A:", answer)

    print("A: Goodbye")
