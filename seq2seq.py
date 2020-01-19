import numpy as np
import tensorflow as tf


np.set_printoptions(threshold=2, linewidth=1000, precision=4)

args = {'train_steps': 409600,
        'epoch': 10,
        'batch_size': 8,
        'evaluations': 1,
        'logs_per_training': 20,
        'eval_steps': 1,
        'time_steps': 15,
        'output_size': 5,
        'embedding_size': 20,
        'sparse_dim': 20,
        'l2': None,
        'learning_rate': 0.0002,
        'num_rnn_nodes': 12,
        'num_rnn_layers': 4,
        'keep_prob': 1,
        'feature_dim': 28,
        'target_dim': 1,
        'gen_num': 2560,
        'model_dir': './model/model_01'}


def input_train_fn(batch_size, epoch, sparse_dim, feature_dim , gen_num):
    #     data_encode = seq_encoder_train.astype(np.float32)
    #     data_decode = seq_decoder_train.astype(np.float32)
    sparse_encode = np.zeros((gen_num, 10, sparse_dim))
    for i in range(gen_num):
        for j in range(10):
            ran_index = np.random.randint(sparse_dim, size=1)
            sparse_encode[i][j][ran_index] = 1
    sparse_encode = sparse_encode.astype(np.float32)
    dense_encode = np.random.rand(gen_num, 10, feature_dim-sparse_dim).astype(np.float32)
    data_encode = np.concatenate((sparse_encode, dense_encode), axis=2)
    data_decode = np.random.rand(gen_num, 5, 1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({'encode': data_encode}, {'decode': data_decode}))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(epoch)
    dataset = dataset.shuffle(1000)
    return dataset


def input_vali_fn(batch_size, feature_dim):
    data_encode = np.random.rand(256, 10, feature_dim).astype(np.float32)
    data_decode = np.random.rand(256, 5, 1).astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({'encode': data_encode}, {'decode': data_decode}))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def input_test_fn(batch_size, feature_dim):
    test_data_encode = np.random.rand(256, 10, feature_dim).astype(np.float32)
    test_data_decode = np.random.rand(256, 5, 1).astype(np.float32)
    #     test_data_encode = build_seq_decoder_data_volume_ratio().astype(np.float32)
    #     test_data_decode = build_seq_decoder_data_volume_ratio().astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({'encode': test_data_encode}, {'decode': test_data_decode}))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def pre_input_fn(batch_size, feature_dim, target_dim):
    #     pre_data_encode = np.random.rand(batch_size,10,feature_dim).astype(np.float32)
    #     pre_data_decode = np.random.rand(batch_size,5,target_dim).astype(np.float32)
    pre_data_encode = build_seq_decoder_data_volume_ratio().astype(np.float32)
    pre_data_decode = build_seq_decoder_data_volume_ratio().astype(np.float32)
    dataset = tf.data.Dataset.from_tensor_slices(({'encode': pre_data_encode}, {'decode': pre_data_decode}))
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset


def _reshape_input(data, batch_size, input_size, embedding_size):
    if data.get_shape().as_list()[1] != input_size:
        raise AssertionError('The data does input_size %d instead of the expected %s' %
                             (data.get_shape().as_list()[1], input_size))

    if data.get_shape().as_list()[2] != embedding_size:
        raise AssertionError('The data does feature_size %d instead of the expected %s' %
                             (data.get_shape().as_list()[2], embedding_size))
    reshaped = tf.reshape(data, [batch_size, -1, embedding_size])


def _get_batch_size(input_data):
    batch_size = input_data.get_shape().as_list()[0]

    if batch_size is None:
        raise AssertionError('Batch size is not known')

    return batch_size


def _loss_weights(batch_size):
    loss_weights = tf.tile([0.5, 0.15, 0.15, 0.1, 0.1], [batch_size])
    loss_weights = tf.reshape(loss_weights, [batch_size, 5, 1])

    return loss_weights


def embedding_layer(input_data_sparse, embedding_size, sparse_dim):
    input_data_sparse = tf.cast(input_data_sparse, dtype=tf.int32)
    embedding = tf.get_variable('embedding_matix', [embedding_size])
    embedded_input = tf.nn.embedding_lookup(embedding, input_data_sparse)

    return embedded_input


def _prepend_go_token(output_data, go_token, dim):
    #     go_tokens = tf.fill([_get_batch_size(output_data),1,feature_dim],go_token)
    #     go_tokens = tf.cast(go_tokens,tf.float32)

    go_tokens = tf.constant(go_token, shape=[_get_batch_size(output_data), 1, dim], dtype=tf.float32)

    #     go_tokens = tf.tile(go_tokens, _get_batch_size((output_data),1,1))

    return tf.concat([go_tokens, output_data], axis=1)


def _make_cell(rnn_size, keep_prob):
    cell = tf.contrib.rnn.GRUCell(rnn_size,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  activation=tf.nn.tanh)
    dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return dropout_cell


def encoding_layer(input_data, input_size, num_rnn_nodes, num_rnn_layers, keep_prob, embedding_size, sparse_dim):
    encoder_cell = tf.contrib.rnn.MultiRNNCell([_make_cell(num_rnn_nodes, keep_prob) for _ in range(num_rnn_layers)])
    print(input_data.shape)
    input_data_dense = input_data[:, :, sparse_dim:]
    input_data_sparse = input_data[:, :, :sparse_dim]
    embedded_input = embedding_layer(input_data_sparse, embedding_size, sparse_dim)
    input_data_concat = tf.concat([embedded_input, input_data_dense], axis=2)
    encoder_output, encoder_state = tf.nn.dynamic_rnn(
        encoder_cell, input_data_concat,
        sequence_length=[input_size] * _get_batch_size(input_data),
        dtype=tf.float32)

    return encoder_output, encoder_state


def decoding_layer(batch_size, num_rnn_nodes, num_rnn_layers, output_size,
                   encoder_state, output_data, target_dim, go_token, regularizer, keep_prob):
    decoder_cell = tf.contrib.rnn.MultiRNNCell([_make_cell(num_rnn_nodes, keep_prob) for _ in range(num_rnn_layers)])

    projection_layer = tf.layers.Dense(units=target_dim,
                                       kernel_initializer=tf.glorot_normal_initializer(),
                                       kernel_regularizer=regularizer)

    training_decoder_output = None
    with tf.variable_scope('decode'):
        if output_data is not None:
            decoder_input = _prepend_go_token(output_data, go_token, target_dim)

            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_input,
                                                                sequence_length=[output_size] * batch_size)
            training_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, training_helper,
                                                               encoder_state, projection_layer)
            training_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                training_decoder, impute_finished=True,
                maximum_iterations=output_size)
    with tf.variable_scope('decode', reuse=tf.AUTO_REUSE):
        start_tokens = tf.constant(go_token, shape=[batch_size, target_dim])
        inference_helper = tf.contrib.seq2seq.InferenceHelper(
            sample_fn=lambda outputs: outputs,
            sample_shape=[target_dim],
            sample_dtype=tf.float32,
            start_inputs=start_tokens,
            end_fn=lambda sample_ids: False)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, inference_helper,
                                                            encoder_state, projection_layer)

        inference_decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
            inference_decoder, impute_finished=True,
            maximum_iterations=output_size)

    return training_decoder_output, inference_decoder_output


def rnn_model_fn(features, labels, mode, params):
    print('-------- Mode:', mode.upper(), '-----------')
    input_size = params['input_size']
    output_size = params['output_size']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    l2_regularization = params['l2_regularization']
    feature_dim = params['feature_dim']
    target_dim = params['target_dim']
    keep_prob = params['keep_prob']
    embedding_size = params['embedding_size']
    sparse_dim = params['sparse_dim']
    if l2_regularization:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regularization)
    else:
        regularizer = None
    num_rnn_layers = params['num_rnn_layers']
    num_rnn_nodes = params['num_rnn_nodes']

    input_data = features['encode']

    if mode != tf.estimator.ModeKeys.PREDICT:
        output_data = labels['decode']
    else:
        output_data = None
    go_token = -1.0

    _, encoder_state = encoding_layer(input_data, input_size, num_rnn_nodes, num_rnn_layers,
                                      keep_prob, embedding_size, sparse_dim)

    training_decoder_output, inference_decoder_output = decoding_layer(batch_size, num_rnn_nodes,
                                                                       num_rnn_layers, output_size,
                                                                       encoder_state, output_data,
                                                                       target_dim, go_token,
                                                                       regularizer, keep_prob)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = inference_decoder_output.rnn_output

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    predictions = training_decoder_output.rnn_output

    #     #loss = tf.losses.mean_squared_error(labels=output_data, predictions=predictions,
    #                                         weights=_loss_weights(args['batch_size']))
    loss = tf.losses.huber_loss(labels=output_data, predictions=predictions,
                                weights=_loss_weights(args['batch_size']),
                                delta=0.2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)


def train_and_evaluate_model(args):
    if args['train_steps'] % args['batch_size']:
        raise ValueError(
            'The number of train steps %d must be a multiple of batch zie %d' %
            (args['train_steps'], args['batch_size']))

    params = {'input_size': args['time_steps'] - args['output_size'],
              'output_size': args['output_size'],
              'batch_size': args['batch_size'],
              'l2_regularization': args['l2'],
              'learning_rate': args['learning_rate'],
              'num_rnn_nodes': args['num_rnn_nodes'],
              'num_rnn_layers': args['num_rnn_layers'],
              'feature_dim': args['feature_dim'],
              'target_dim': args['target_dim'],
              'keep_prob': args['keep_prob'],
              'embedding_size': 20,
              'sparse_dim': 20,
              }
    log_step_count_steps = max(1, args['train_steps'] / args['batch_size'] /
                               args['evaluations'] // args['logs_per_training'])
    estimator = tf.estimator.Estimator(
        model_dir=args['model_dir'],
        model_fn=rnn_model_fn,
        params=params,
        config=tf.estimator.RunConfig(log_step_count_steps=log_step_count_steps))
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_train_fn(args['batch_size'], args['epoch'], args['sparse_dim'], args['feature_dim'],
                                        args['gen_num']),
        max_steps=args['train_steps'] // args['batch_size'])
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_vali_fn(args['batch_size'], args['feature_dim']),
                                      steps=args['eval_steps'])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    return estimator


def _calcu_score(array1, array2):
    sum_score = 0
    weight = [0.5, 0.15, 0.15, 0.1, 0.1]
    for i in range(5):
        score = weight[i] * (1 - abs(array2[i] - array1[i]) / array1[i])
        sum_score = sum_score + score

    if sum_score >= 0:
        return sum_score
    else:
        return 0


def get_score(expected, predicted):
    expected = np.transpose(expected, axes=[0, 2, 1])
    predicted = np.transpose(predicted, axes=[0, 2, 1])
    assert expected.shape == predicted.shape
    [dim1, dim2, _] = expected.shape
    score_volume = []
    score_hold = []
    for i in range(dim1):
        score_volume.append(_calcu_score(expected[i, 0], predicted[i, 0]))
    #         score_hold.append(_calcu_score(expected[i,1],predicted[i,1]))
    return np.asarray(score_volume)


def make_predictions(args, estimator):
    predict_results = list(estimator.predict(input_fn=lambda: input_test_fn(args['batch_size'], args['feature_dim'])))
    predict_results = np.asarray(predict_results)
    test_dataset = input_test_fn(args['batch_size'], args['feature_dim'])
    test_iterator = test_dataset.make_one_shot_iterator()
    next_element = test_iterator.get_next()

    with tf.Session() as sess:
        try:
            batch_index = 0
            average_score = []
            j = 1
            TA_score = []
            SR_score = []
            MA_score = []
            CF_score = []
            ZC_score = []
            AP_score = []

            for i in predict_results:
                test_data = sess.run(next_element)
                batch_predict = predict_results[batch_index:batch_index + args['batch_size']]
                # print('expected:',test_data[1]['decode'])
                zip_1 = zip(test_data[1]['decode'].tolist(),
                            predict_results[batch_index:batch_index + args['batch_size']].tolist())
                zip_2 = zip(zip_1, get_score(test_data[1]['decode'], batch_predict).tolist())

                for k in list(zip_2):
                    print(k)
                    print('******************')

                j = j + 1
                average_score.append(np.average(get_score(test_data[1]['decode'], batch_predict)))
                batch_index = batch_index + args['batch_size']
        except tf.errors.OutOfRangeError:
            print('average score total', np.average(average_score))
            print('TA average score', np.average(TA_score))
            print('SR average score', np.average(SR_score))
            print('MA average score', np.average(MA_score))
            print('CF average score', np.average(CF_score))
            print('ZC average score', np.average(ZC_score))
            print('AP average score', np.average(AP_score))
    return predict_results


def _mse(expected, predicted):
    return ((np.asarray(expected) - np.asarray(predicted) ** 2)).mean()


def main(argv):
    print('Arguments', args)
    estimator = train_and_evaluate_model(args)
    make_predictions(args, estimator)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)