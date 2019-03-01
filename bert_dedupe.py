""" Dedupe Domains:
    - homesteady
    - oureverydaylife
    - ourpastimes
    - gardenguides
+ Find similarity using BERT and write output to csv with columns ('url,similar_url,cosine_similarity')
+ Code reused from tensorflow BERT tutorial
    - dependent on run_classifier.py, tokenization.py
"""
from prepare_redirect import prepare_redirect_list
import fire
import tensorflow as tf
import tensorflow_hub as hub
import run_classifier
import tokenization
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time


BERT_MODULE = 'https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1'
MAX_SEQ_LENGTH = 64
BATCH_SIZE = 32


def create_tokenizer_from_hub_module():
    """ Gets vocab file and casing from Hub module """
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODULE, trainable=True)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def get_training_data(domain):
    """ Gets data in the form that BERT accepts to embed """
    processor = run_classifier.Ehow_Dedupe_Processor()

    tokenizer = create_tokenizer_from_hub_module()

    train_examples, docs = processor.get_train_examples(domain)

    print("done preprocessing...")
    train_features = run_classifier.convert_examples_to_features(examples=train_examples, label_list=None,
                                                                max_seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)


    return train_features, docs


def get_bert_embedding(bert_module, input_ids, input_mask, segment_ids):
    """ Gets embedding of size [num_documents, embedding_dim] based on input data
        args:
            input_ids: size [num_documents, max_seq_length]
            input_mask: size [num_documents, max_seq_length]
            segment_ids: size [num_documents, max_seq_length]
        returns:
            output: matrix of size [num_documents, embedding_dim]
    """
    # bert_module = hub.Module(BERT_MODULE, trainable=True)
    bert_inputs = dict(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids
    )
    bert_outputs = bert_module(
        inputs=bert_inputs,
        signature="tokens",
        as_dict=True
    )

    output = bert_outputs["pooled_output"]
    return output


def similarity(embedding_matrix, index_to_id_mapping):
    """ Returns similar documents """
    sim_matrix = cosine_similarity(embedding_matrix)


    most_similar = np.argmax(sim_matrix, axis=1)
    scores = [row[most_similar[i]] for i, row in enumerate(sim_matrix)]

    for i, max_idx in enumerate(most_similar):
        print(f"{index_to_id_mapping[i]} is most similar to {index_to_id_mapping[max_idx]} with score {scores[i]}")







def get_embedding_matrix(domain):
    train_features, docs = get_training_data(domain)
    input_ids = []
    input_mask = []
    segment_ids = []
    for train_feature in train_features:
        input_ids.append(train_feature.input_ids)
        input_mask.append(train_feature.input_mask)
        segment_ids.append(train_feature.segment_ids)

    print("done getting features...")
    embeddings = None
    embedding_array = []
    with tf.Graph().as_default():
        with tf.Session() as sess:
            bert_module = hub.Module(BERT_MODULE)

            writer = tf.summary.FileWriter('tensorboard')
            writer.add_graph(tf.get_default_graph())
            writer.flush()

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            print("starting embedding...")
            total_num_documents = len(input_ids)
            for i in range(total_num_documents//BATCH_SIZE + 1):
                start = time.time()
                try:
                    batch_input_ids = input_ids[i*BATCH_SIZE:i*BATCH_SIZE + BATCH_SIZE]
                    batch_input_mask = input_mask[i*BATCH_SIZE:i*BATCH_SIZE + BATCH_SIZE]
                    batch_segment_ids = segment_ids[i*BATCH_SIZE:i*BATCH_SIZE + BATCH_SIZE]
                except IndexError: # for last batch
                    batch_input_ids = input_ids[i*BATCH_SIZE:]
                    batch_input_mask = input_mask[i*BATCH_SIZE:]
                    batch_segment_ids = segment_ids[i*BATCH_SIZE:]

                bert_embeddings = get_bert_embedding(bert_module, batch_input_ids, batch_input_mask, batch_segment_ids)

                batch_embeddings = sess.run(bert_embeddings)

                embedding_array.append(batch_embeddings)
                end = time.time()
                print(batch_embeddings.shape)
                print(f"batch {i}, time taken: {end-start}")

    embeddings = np.concatenate(embedding_array, 0)

    return embeddings, docs



def main(domain):
    embedding_matrix, docs = get_embedding_matrix(domain)

    index_to_id_mapping = {i:doc for i, doc in enumerate(docs)}


    similarity(embedding_matrix, index_to_id_mapping)







if __name__ == '__main__':
    fire.Fire(main)
