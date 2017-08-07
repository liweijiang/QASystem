from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

from utils.data_reader import read_data, load_glove_embeddings
from qa_data import sentence_to_token_ids
from utils.data_tool import add_space_between_word_char_sentence, find_phrase_given_span
from tensorflow.python.saved_model import builder as saved_model_builder

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.20, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 24, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 1, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("encoder_state_size", 100, "Size of each encoder model layer.")
tf.app.flags.DEFINE_integer("decoder_state_size", 100, "Size of each decoder model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("single_question", "What is density?", "Question regarding a context paragraph.")
tf.app.flags.DEFINE_string("single_context", "The density, or more precisely, the volumetric mass density, of a substance is its mass per unit volume.", "Context paragraph to ask a question.")
tf.app.flags.DEFINE_string("mode", "single", "Mode of session (default: train).")
tf.app.flags.DEFINE_string("export_path_base", "qasystem_export", "Export path base name.")
tf.app.flags.DEFINE_integer("model_version", 1, "Model version to export.")

tf.app.flags.DEFINE_integer("question_maxlen", 30, "Max length of question (default: 30")
tf.app.flags.DEFINE_integer("context_maxlen", 400, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_integer("n_features", 1, "Number of features for each position in the sentence.")
tf.app.flags.DEFINE_integer("log_batch_num", 100, "Number of batches to write logs on tensorboard.")
tf.app.flags.DEFINE_integer("decoder_hidden_size", 100, "Number of decoder_hidden_size.")
tf.app.flags.DEFINE_boolean("QA_ENCODER_SHARE", True, "QA_ENCODER_SHARE weights.")
tf.app.flags.DEFINE_boolean("tensorboard", False, "Write tensorboard log or not.")
tf.app.flags.DEFINE_boolean("RE_TRAIN_EMBED", False, "Max length of the context (default: 400)")
tf.app.flags.DEFINE_string("debug_train_samples", None, "number of samples for debug (default: None)")
tf.app.flags.DEFINE_float("ema_weight_decay", 0.999, "exponential decay for moving averages ")
tf.app.flags.DEFINE_integer("evaluate_sample_size", 10, "number of samples for evaluation (default: 400)")
tf.app.flags.DEFINE_integer("model_selection_sample_size", 10, "number of samples for selecting best model (default: 1000)")
tf.app.flags.DEFINE_integer("window_batch", 3, "window size / batch size")

FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = train_dir
    #global_train_dir = '/tmp/cs224n-squad-train'
    #if os.path.exists(global_train_dir):
    #    os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    #print('source: ',os.path.abspath(train_dir))
    #print('dst: ', global_train_dir)
    #os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

# create a single evaluation data with format: [question, len(question), context, len(context)]
# question: question string // context: context string // vocab: vocabulary
def preprocess_single_eval_data(question, context, vocab):
    question = add_space_between_word_char_sentence(question)
    context = add_space_between_word_char_sentence(context)
    question_id = sentence_to_token_ids(question, vocab)
    context_id = sentence_to_token_ids(context, vocab)

    return [question_id, len(question_id), context_id, len(context_id), [0, 0]], question, context

def main(_):

    #dataset = read_data(FLAGS.data_dir, small_dir=None, small_val=None, \
    #    debug_train_samples=FLAGS.debug_train_samples, debug_val_samples=100, context_maxlen=FLAGS.context_maxlen)
    dataset = read_data(FLAGS.data_dir)
    if FLAGS.context_maxlen is None:
        FLAGS.context_maxlen = dataset['context_maxlen']
    if FLAGS.question_maxlen is None:
        FLAGS.question_maxlen = dataset['question_maxlen']

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = load_glove_embeddings(embed_path)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    qa = QASystem(embeddings, FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    # print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    gpu_options = tf.GPUOptions()
    #gpu_options.allow_growth=True

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        if FLAGS.mode == 'train':
            logging.info("Training model ...")
            save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
            qa.train(sess, dataset, save_train_dir, rev_vocab)
            # qa.export_model(sess, FLAGS.export_path_base, dataset, rev_vocab)

        elif FLAGS.mode == 'validate':
            logging.info("Validating model ...")
            save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
            f1, em = qa.evaluation(sess, dataset, save_train_dir, rev_vocab)

        elif FLAGS.mode == 'export':
            logging.info("Exporting model ...")
            qa.export_model(sess, FLAGS.export_path_base, dataset, rev_vocab)

        elif FLAGS.mode == 'single':
            while True:
                single_context = raw_input("Please enter a context paragraph: ")
                single_question = raw_input("Please enter a question: ")

                if (len(single_context) == 0) or (len(single_question) == 0):
                    pass
                else:
                    evaluation_data, raw_question_data, raw_context_data = preprocess_single_eval_data(single_question, single_context, vocab)
                    predict_answer, (start, end) = qa.evaluation_single(sess, dataset, rev_vocab, evaluation_data)
                    print('*' * 50 )
                    print('|| Predicted Answer: || ' + predict_answer)
                    print('|| Answer Span: || ' + str(start) + ',' + str(end))
                    predict_answer = find_phrase_given_span(raw_context_data, (start, end))
                    print('|| Raw Predicted Answer: || ' + predict_answer)

        elif FLAGS.mode == 'export':
            qa.export_model(FLAGS.export_path_base)


if __name__ == "__main__":
    tf.app.run()
