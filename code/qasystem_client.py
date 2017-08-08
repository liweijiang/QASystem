# Filename: qasystem_client.py
# Author: Liwei Jiang
# Date: 2017/08/08
# Usage: Used to request predicted outputs of the qasystem
#        model given certain inputs from TensorFlow Server.
# Reference: https://github.com/yolandawww/QASystem

from __future__ import print_function

# This is a placeholder for a Google-internal import.
from grpc.beta import implementations
import tensorflow as tf
import numpy

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from qa_data import sentence_to_token_ids
from data_tool import add_space_between_word_char_sentence, find_phrase_given_span

tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def initialize_vocab(vocab_path):
    """ Initialize the vocab dictionary. """
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def get_best_span(start_logits, end_logits, context_ids):
    """ Get the best span given the predicted probability and the contex paragraph. """
    start_sentence_logits = []
    end_sentence_logits = []
    new_start_sentence = []
    new_end_sentence = []
    for i, c_id in enumerate(context_ids):
        new_start_sentence.append(start_logits[i])
        new_end_sentence.append(end_logits[i])
        if c_id == 6: # dot id, represents the end of a sentence
            start_sentence_logits.append(new_start_sentence)
            end_sentence_logits.append(new_end_sentence)
            new_start_sentence = []
            new_end_sentence = []
    if len(new_start_sentence) > 0:
        start_sentence_logits.append(new_start_sentence)
        end_sentence_logits.append(new_end_sentence)

    best_word_span = (0, 0)
    best_sent_idx = 0
    argmax_j1 = 0
    max_val = start_logits[0] + end_logits[0]
    for f, (ypif, yp2if) in enumerate(zip(start_sentence_logits, end_sentence_logits)):
        argmax_j1 = 0
        for j in range(len(ypif)):
            val1 = ypif[argmax_j1]
            if val1 < ypif[j]:
                val1 = ypif[j]
                argmax_j1 = j

            val2 = yp2if[j]
            if val1 + val2 > max_val:
                best_word_span = (argmax_j1, j)
                best_sent_idx = f
                max_val = val1 + val2
    len_pre = 0
    for i in range(best_sent_idx):
        len_pre += len(start_sentence_logits[i])
    best_word_span = (len_pre + best_word_span[0], len_pre + best_word_span[1])
    return best_word_span, max_val


def preprocess_single_eval_data(question, context, vocab):
    """
        Preprocess the single evaluation data.
        Tokenize the raw context and question text.
        Get the id of the raw context and question.
        Arrange them into a list: [question, len(question), context, len(context), answer]
    """
    question = add_space_between_word_char_sentence(question)
    context = add_space_between_word_char_sentence(context)
    question_id = sentence_to_token_ids(question, vocab)
    context_id = sentence_to_token_ids(context, vocab)

    return [question_id, len(question_id), context_id, len(context_id), [0, 0]], question, context


def main(_):
    """ Request predicted results from the TensorFlow Server. """
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'qasystem'
    request.model_spec.signature_name = 'predict_answer'
    vocab, rev_vocab = initialize_vocab("tensorflow_serving/example/vocab.dat")

    inputs_context_raw = 'The density at all points of a homogeneous object equals its total mass divided by its total volume. The mass is normally measured with a scale or balance; the volume may be measured directly (from the geometry of the object) or by the displacement of a fluid . To determine the density of a liquid or a gas, a hydrometer, a dasymeter or a Coriolis flow meter may be used, respectively. Similarly, hydrostatic weighing uses the displacement of water due to a submerged object to determine the density of the object.'
    input_question_raw = 'What to use to determine the density of a liquid or a gas?'

    # [question, len(question), context, len(context), answer]
    evaluation_data, raw_question_data, raw_context_data = preprocess_single_eval_data(input_question_raw, inputs_context_raw, vocab)

    print('------- Raw Question: ')
    print(input_question_raw)
    print('------- Raw Context: ')
    print(inputs_context_raw)
    # print('------- Evaluation Data: ')
    # print(evaluation_data)

    serving_inputs_context = [evaluation_data[2]]
    serving_inputs_context_mask = [[]]
    for i in range(evaluation_data[3]):
        serving_inputs_context_mask[0].append(True)
    serving_inputs_question = [evaluation_data[0]]
    serving_inputs_question_mask = [[]]
    for i in range(evaluation_data[1]):
        serving_inputs_question_mask[0].append(True)
    serving_inputs_JX = evaluation_data[3]
    serving_inputs_JQ = evaluation_data[1]
    serving_inputs_dropout = 1.0

    # print('-------------- serving_inputs_context --------------')
    # print(serving_inputs_context)
    # print('-------------- serving_inputs_context_mask --------------')
    # print(serving_inputs_context_mask)
    # print('-------------- serving_inputs_question --------------')
    # print(serving_inputs_question)
    # print('-------------- serving_inputs_question_mask --------------')
    # print(serving_inputs_question_mask)
    # print('-------------- serving_inputs_JX --------------')
    # print(serving_inputs_JX)
    # print('-------------- serving_inputs_JQ --------------')
    # print(serving_inputs_JQ)
    # print('-------------- serving_inputs_dropout --------------')
    # print(serving_inputs_dropout)

    request.inputs['context'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_context, shape=None))
    request.inputs['context_mask'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_context_mask, shape=None))
    request.inputs['question'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_question, shape=None))
    request.inputs['question_mask'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_question_mask, shape=None))
    request.inputs['JX'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_JX, dtype=tf.int32, shape=[]))
    request.inputs['JQ'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_JQ, dtype=tf.int32, shape=[]))
    request.inputs['dropout'].CopyFrom(
        tf.contrib.util.make_tensor_proto(serving_inputs_dropout, dtype=tf.float32, shape=[]))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    result_future = stub.Predict.future(request, 10.0)  # 10 secs timeout

    span_start = numpy.array([numpy.array(result_future.result().outputs['span_start'].float_val)])
    span_end = numpy.array([numpy.array(result_future.result().outputs['span_end'].float_val)])
    context_batch = numpy.array([numpy.array(serving_inputs_context_mask[0])])

    best_spans, scores = zip(*[get_best_span(si, ei, ci) for si, ei, ci in zip(span_start, span_end, context_batch)])

    # print('-------------- best_spans --------------')
    # print(type(best_spans))
    # print(best_spans)
    # print(type(best_spans))
    #
    # print('-------------- raw_context_data --------------')
    # print(type(raw_context_data))
    # print(raw_context_data)
    # print(len(raw_context_data))

    predict_answer = find_phrase_given_span(raw_context_data, best_spans[0])

    print('-------------- Predicted Answer --------------')
    print(predict_answer)

if __name__ == '__main__':
    tf.app.run()
