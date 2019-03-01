from get_data import get_data
import tensorflow as tf
import tensorflow_hub as hub
import fire
from sklearn.metrics.pairwise import cosine_similarity
import time

BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128



def get_elmo_embedding(list_sentence):
    """ Returns embedding for documents """
    list_sentence = list(map(lambda x: ' '.join(x.split()[:MAX_SEQ_LENGTH]), list_sentence))
    embedding_array = []
    with tf.Graph().as_default():
        elmo = hub.Module("https://tfhub.dev/google/elmo/2")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            total_num_documents = len(list_sentence)
            for i in range(total_num_documents//BATCH_SIZE + 1):
                start = time.time()
                try:
                    batch_sentence = list_sentence[i*BATCH_SIZE:i*BATCH_SIZE + BATCH_SIZE]
                except IndexError: # for last batch
                    batch_sentence = list_sentence[i*BATCH_SIZE:]
                embeddings = elmo(batch_sentence, signature="default", as_dict=True)["default"]

                batch_embeddings = sess.run(embeddings)

                embedding_array.append(batch_embeddings)
                end = time.time()
                print(batch_embeddings.shape)
                print(f"batch {i}, time taken: {end-start}")

    embedding_matrix = np.concatenate(embedding_array, 0)

    return embedding_matrix






def main(domain):
    docs = get_data(domain)
    texts = list(docs.values())
    embedding_matrix = get_elmo_embedding(texts)

    print(embedding_matrix.shape)
    sim = cosine_similarity(embedding_matrix)

    print(sim)



if __name__ == '__main__':
    fire.Fire(main)
