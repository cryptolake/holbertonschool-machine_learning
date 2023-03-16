# QA bot

We create a QA bot using transfer learning with a BERT model.

## Models

- GPT-2: gpt-2 influenced gpt-3 and bert in which it was concluded
that the best way to train a language model is through unsupervised
training.

- Bert: a bidirectional transforfmer based model, it was trained by
going through a large corpus picking a word masking it and generating
it by what came before it and after it this is called MLM,
the other technique is to predict whether two sentences in a pair
are adjacent or not this helps with coherencec and it was called NSP. 
that's the unsupervised learning part, then it was fine tuned to a specific domain.

- Transfomer-XL: In Transformer-XL they mitigated both the long term limitation
of both vanilla transfomers and RNNs even though vanilla Transfomers were set to 
solve this there was the limitation of the fixed length context window, first
transformer-XL concatenates the previous hidden state with the current one, second
it uses a different positional encoding that instead of an absolute measure of
position, it uses a relative one.

- GPT-3: gpt-3 introduced zero shot training in which we only train
the model in an unsupervised way so there will be no need for task specific
fine-tuning, this is achieved by utilizing a much bigger model:
175B parameters.

- XLNet: XLNet is the combination of Bert and Transformer-XL it uses the bidirectional
nature of bert but without its masking token it instead performs all the permutation of
the context word pretraining (it computes the word given the previous words in all orders of
computation) this will have the bidirectional effects of Bert, as for transformer-XL it
uses its relative encoding strategy and its concatenation of previous hidden states.

