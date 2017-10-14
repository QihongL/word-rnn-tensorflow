import numpy as np

def get_top_k_probs_and_indices(prob_table, k):
  '''
  prob_table: n_words x n_tokens: probability matrix
  k: a positive integer
  '''
  n_words, n_tokens = np.shape(prob_table)
  if k < 1 or k > n_tokens:
    raise ValueError('0 < k <= smaller than num_tokens')
  # loop over rows (words)
  for wi in range(n_words):
    # get the i-th row
    prob_vec_wi = prob_table[wi, :]
    # ge the sorting index of the probability
    sorting_idx_descending = np.argsort(prob_vec_wi)[::-1]
    prob_vec_wi = prob_vec_wi[sorting_idx_descending]
    # collect the top k probability, and the top k indices
    if wi == 0:
      prob_table_top_k = prob_vec_wi[:k]
      sort_table_top_k = sorting_idx_descending[:k]
    else:
      prob_table_top_k = np.vstack([prob_table_top_k, prob_vec_wi[:k]])
      sort_table_top_k = np.vstack([sort_table_top_k, sorting_idx_descending[:k]])
  return prob_table_top_k, sort_table_top_k

