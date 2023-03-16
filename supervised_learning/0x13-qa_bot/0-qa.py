#!/usr/bin/env python3
"""
QA bot using pretrained bert with local documents
as dictionary.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
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
    type_ids = [0] * (1 + len(tok_ref) + 1) + [1] * (len(tok_que) + 1)
    word_ids, mask, type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
                                   (word_ids, mask, type_ids))
    outputs = model([word_ids, mask, type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    # answer2 = tokenizer.convert_tokens_to_string(outputs)
    return answer
