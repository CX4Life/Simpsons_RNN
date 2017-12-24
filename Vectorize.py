"""Convert integer-based transcript to matrix of 300-dimensional vectors."""

import os
import glob
import json
import math
import datetime as dt
import tensorflow as tf
import numpy as np
import collections
import random

__author__ = ['Tim Woods',
              'adventuresinML']
__copyright__ = 'Copyright (c) 2017 Tim Woods'
__license__ = 'MIT'

CLEANED_DIR = 'cleaned/'

def get_all_cleaned_scripts():
    """Hey now do the hustle. """
    pwd = os.getcwd()
    os.chdir(CLEANED_DIR)
    all_files = glob.glob("*.txt")
    os.chdir(pwd)
    return all_files


def update_running_script(data, new_script):
    for elem in new_script:
        data.append(elem)
    return data


num_embeddings = get_all_cleaned_scripts()
reverse_dictionary = json.load(open('lookup_dict.json', 'rb'))
reverse_dictionary['0'] = 'UNK'
data = list()
for emb in num_embeddings:
    a_script = json.load(open(CLEANED_DIR + emb, 'rb'))
    data = update_running_script(data, a_script)

dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))


data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    """Copied from GitHub account adventuresinML"""
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


batch_size = 128
embedding_size = 300  # Dimension of the embedding vector.
skip_window = 2       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.
vocabulary_size = 10000

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Look up embeddings for inputs.
  embeddings = tf.Variable(
      tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)

  # Construct the variables for the softmax
  weights = tf.Variable(
      tf.truncated_normal([embedding_size, vocabulary_size],
                          stddev=1.0 / math.sqrt(embedding_size)))
  biases = tf.Variable(tf.zeros([vocabulary_size]))
  hidden_out = tf.transpose(tf.matmul(tf.transpose(weights), tf.transpose(embed))) + biases

  # convert train_context to a one-hot format
  train_one_hot = tf.one_hot(train_context, vocabulary_size)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hidden_out, labels=train_one_hot))

  # Construct the SGD optimizer using a learning rate of 1.0.
  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.global_variables_initializer()


def run(graph, num_steps):
    with tf.Session(graph=graph) as session:
      # We must initialize all variables before we use them.
      init.run()
      print('Initialized')

      average_loss = 0
      for step in range(num_steps):
        batch_inputs, batch_context = generate_batch(data,
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
          if step > 0:
            average_loss /= 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step ', step, ': ', average_loss)
          average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[str(valid_examples[i])]
            top_k = 8  # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[str(nearest[k])]
              log_str = '%s %s,' % (log_str, close_word)
            print(log_str)
      final_embeddings = normalized_embeddings.eval()

num_steps = 100
softmax_start_time = dt.datetime.now()
run(graph, num_steps=num_steps)
softmax_end_time = dt.datetime.now()
print("Softmax method took {} minutes to run 100 iterations".format((softmax_end_time-softmax_start_time).total_seconds()))

with graph.as_default():

    # Construct the variables for the NCE loss
    nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    nce_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                       biases=nce_biases,
                       labels=train_context,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    # Add variable initializer.
    init = tf.global_variables_initializer()

num_steps = 50000
nce_start_time = dt.datetime.now()
run(graph, num_steps)
nce_end_time = dt.datetime.now()
print("NCE method took {} minutes to run 100 iterations".format((nce_end_time-nce_start_time).total_seconds()))
