from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import argparse
import time
import os
from six.moves import cPickle
from utils import TextLoader
from utils_q import *
from model import Model
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train_dir', type=str, default='data/amy_train',
                       help='data directory containing input.txt')
    parser.add_argument('--data_test_dir', type=str, default='data/amy_test',
                        help='data directory containing input.txt')
    parser.add_argument('--input_encoding', type=str, default=None,
                       help='character encoding of input.txt')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='directory containing tensorboard logs')
    parser.add_argument('--save_dir', type=str, default='save',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=32,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    # parser.add_argument('--batch_size', type=int, default=2,
    #                    help='minibatch size')
    # parser.add_argument('--seq_length', type=int, default=5,
    #                    help='RNN sequence length')
    parser.add_argument('--batch_size', type=int, default=12,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=41,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=10000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                       help='decay rate for rmsprop')
    parser.add_argument('--gpu_mem', type=float, default=1,
                       help='%% of gpu memory to be allocated to this process. Default is 66.6%%')
    parser.add_argument('--init_from', type=str, default=None,
                       help="""continue training from saved model at this path. Path must contain files saved by previous training process:
                            'config.pkl'        : configuration;
                            'words_vocab.pkl'   : vocabulary definitions;
                            'checkpoint'        : paths to model file(s) (created by tf).
                                                  Note: this file contains absolute paths, be careful when moving files around;
                            'model.ckpt-*'      : file(s) with model definition (created by tf)
                        """)
    args = parser.parse_args()
    run_model(args)

def run_model(args, test = True):
    data_loader = TextLoader(args.data_train_dir, args.batch_size, args.seq_length, args.input_encoding)
    args.vocab_size = data_loader.vocab_size

    # check compatibility if training is continued from previously saved model
    if args.init_from is not None:
        # check if all necessary files exist
        assert os.path.isdir(args.init_from)," %s must be a path" % args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"config.pkl")),"config.pkl file does not exist in path %s"%args.init_from
        assert os.path.isfile(os.path.join(args.init_from,"words_vocab.pkl")),"words_vocab.pkl.pkl file does not exist in path %s" % args.init_from
        ckpt = tf.train.get_checkpoint_state(args.init_from)
        assert ckpt,"No checkpoint found"
        assert ckpt.model_checkpoint_path,"No model path found in checkpoint"

        # open old config and check if models are compatible
        with open(os.path.join(args.init_from, 'config.pkl'), 'rb') as f:
            saved_model_args = cPickle.load(f)
        need_be_same=["model","rnn_size","num_layers","seq_length"]
        for checkme in need_be_same:
            assert vars(saved_model_args)[checkme]==vars(args)[checkme],"Command line argument and saved model disagree on '%s' "%checkme

        # open saved vocab/dict and check if vocabs/dicts are compatible
        with open(os.path.join(args.init_from, 'words_vocab.pkl'), 'rb') as f:
            saved_words, saved_vocab = cPickle.load(f)
        assert saved_words==data_loader.words, "Data and loaded model disagree on word set!"
        assert saved_vocab==data_loader.vocab, "Data and loaded model disagree on dictionary mappings!"

    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.words, data_loader.vocab), f)

    model = Model(args)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(args.log_dir)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        train_writer.add_graph(sess.graph)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        # restore model
        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for e in range(model.epoch_pointer.eval(), args.num_epochs):
            sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            state = sess.run(model.initial_state)
            speed = 0
            if args.init_from is None:
                assign_op = model.epoch_pointer.assign(e)
                sess.run(assign_op)
            if args.init_from is not None:
                data_loader.pointer = model.batch_pointer.eval()
                args.init_from = None
            for b in range(data_loader.pointer, data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, _, _ = sess.run([merged, model.cost, model.final_state,
                                                             model.train_op, model.inc_batch_pointer_op], feed)
                train_writer.add_summary(summary, e * data_loader.num_batches + b)
                speed = time.time() - start
                if (e * data_loader.num_batches + b) % args.batch_size == 0:
                    print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                        .format(e * data_loader.num_batches + b,
                                args.num_epochs * data_loader.num_batches,
                                e, train_loss, speed))
                if (e * data_loader.num_batches + b) % args.save_every == 0 \
                        or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))
        train_writer.close()

        if test:
            # todo: this is very hacky... change it later
            test_data_loader = TextLoader(args.data_test_dir, args.batch_size, args.seq_length, args.input_encoding)

            # loop over the entire data set and generate the probabilities of the next word
            first_batch = True
            for b in range(test_data_loader.pointer, test_data_loader.num_batches):
                x, y = test_data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y, model.initial_state: state,
                        model.batch_time: speed}
                summary, train_loss, state, probs, _, _ = sess.run([merged, model.cost, model.final_state, model.probs,
                                                                model.train_op, model.inc_batch_pointer_op], feed)

                # save probability vectors along with the text
                # print(np.shape(probs))
                # get probability and indices for the top k predictions
                k = 100
                prob_table_top_k, sorting_idx_table_top_k = get_top_k_probs_and_indices(probs, k)

                # collect info
                if first_batch:
                    PROBS = prob_table_top_k
                    IDX   = sorting_idx_table_top_k
                    first_batch = False
                else:
                    PROBS = np.vstack([PROBS, prob_table_top_k])
                    IDX   = np.vstack([IDX,   sorting_idx_table_top_k])

            # save the probability table and indices
            print(np.shape(PROBS))
            print('whole seq length = %d' % test_data_loader.full_text_len)
            np.savez(os.path.join(args.data_test_dir, 'probs'), prob_table=PROBS, idx_table=IDX)


if __name__ == '__main__':
    main()
