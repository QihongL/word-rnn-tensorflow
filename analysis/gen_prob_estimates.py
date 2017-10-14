import numpy as np
import pickle
import os

# spec paths

data_name = 'toy'
out_pred_fname = 'predictions'
k = 3

# get file names for training and test set
data_test_root = data_name + '_test'
data_train_root = data_name + '_train'
# construct all paths
dict_test_path = os.path.join('..', 'data', data_test_root, 'vocab.pkl')
dict_prob_path = os.path.join('..', 'data', data_train_root, 'vocab.pkl')
text_path = os.path.join('..', 'data', data_test_root, 'data.npy')
prob_path = os.path.join('..', 'data', data_test_root, 'probs.npy')
out_pred_path = os.path.join('..', 'data', data_test_root, out_pred_fname + '.txt')


def get_top_k_preds(k, prob_vec, prob_dict):
  idx_top_k = prob_vec.argsort()[-k:][::-1]
  prob_top_k = prob_vec[idx_top_k]
  return idx_top_k, prob_top_k


# print test set text
def print_test_set(n, text, vocab_test_dict):
  temp = ''
  for i in range(n):
    temp += vocab_test_dict[text[i]] + ' '
  print(temp)


def print_most_likely_preds(n, prob, vocab_prob_dict):
  pred = ''
  for i in range(n):
    idx_top_k, prob_top_k = get_top_k_preds(k, prob[i, :], vocab_prob_dict)
    most_likely_word = vocab_prob_dict[idx_top_k[0]]
    pred += most_likely_word + ' '
  print(pred)


# load the dict that translates test set vs. probability estimates
with open(dict_test_path, 'rb') as f:
  vocab_test_dict = pickle.load(f)
with open(dict_prob_path, 'rb') as f:
  vocab_prob_dict = pickle.load(f)

# load test set text and probability estimates
text = np.load(text_path)
prob = np.load(prob_path)

# get min (pred set, test set) size
n_test_words = np.shape(text)[0]
n_preds = np.shape(prob)[0]
n = np.min([n_test_words, n_preds])

# print out the prediction
with open(out_pred_path, "w") as out_pred_file:
  # loop over all words
  matches = 0
  out_pred_file_text = ''
  for i in range(n):
    idx_top_k, prob_top_k = get_top_k_preds(k, prob[i, :], vocab_prob_dict)
    words_top_k = []
    for j in range(k):
      words_top_k.append(vocab_prob_dict[idx_top_k[j]])

    if i < n - 1:
      if vocab_test_dict[text[i+1]] in words_top_k:
        matches +=1

    curr_word = vocab_test_dict[text[i]]
    out_pred_file_text += curr_word + ' ' + str(words_top_k) + ' ' + str(prob_top_k) + '\n'

  out_pred_file.write(out_pred_file_text)
  print('top %d words match rate = %f' % (k, matches / (n-1)))
