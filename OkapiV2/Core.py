from OkapiV2 import Losses, Accuracies, Optimizers, Initializers
import theano
import theano.tensor as T
import numpy as np
import sklearn as sk
import sys
import pickle
import time


def atleast_4d(x):
    if x.ndim < 4:
        return np.expand_dims(np.atleast_3d(x), axis=3).astype('float32')
    else:
        return x.astype('float32')


def save_model(model, filename='okapi_model.pk'):
    sys.setrecursionlimit(10000)
    file = open(filename, 'wb')
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def load_model(filename='okapi_model.pk'):
    file = open(filename, 'rb')
    model = pickle.load(file)
    file.close()
    return model


def make_batches(x, y, batch_size=128, shuffle=True, nest=True):
    for i in range(len(x)):
        x[i] = atleast_4d(x[i])
    y = atleast_4d(y)
    num_batches = (y.shape[0] // batch_size)
    if y.shape[0] % batch_size is not 0:
        num_batches += 1
    if shuffle:
        shuffled_arrays = sk.utils.shuffle(*x, y)
        x = shuffled_arrays[:len(x)]
        y = shuffled_arrays[-1]
    x_batches_list = []
    for i in range(len(x)):
        x_batches_list.append(np.array_split(x[i], num_batches))
    if nest:
        x_batches = []
        for i in range(num_batches):
            x_batch = []
            for x_input in x_batches_list:
                x_batch.append(x_input[i])
            x_batches.append(x_batch)
    else:
        x_batches = x_batches_list
    y_batches = np.array_split(y, num_batches)
    return x_batches, y_batches, num_batches


class Branch():
    def __init__(self):
        self.inputs = []
        self.merge_mode = 'flat_append'
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def add_input(self, input):
        self.inputs.append(input)

    def get_init_params(self, init_params_list):
        self.get_output_dim()
        init_params = []
        prev_output_dim = self.input_shape
        for layer in self.layers:
            output_dim = layer.get_output_dim(prev_output_dim)
            layer_inits = layer.get_init_params(prev_output_dim)
            if layer_inits is not None:
                layer_inits = [i.astype('float32') for i in layer_inits]
            init_params.append(layer_inits)
            prev_output_dim = output_dim
        init_params_list += [init_params]
        for input in self.inputs:
            if not isinstance(input, np.ndarray):
                init_params_list = input.get_init_params(init_params_list)
        return init_params_list

    def get_output_dim(self):
        shapes = []
        if len(self.inputs) > 1 and self.merge_mode is 'flat_append':
            for input in self.inputs:
                if isinstance(input, np.ndarray):
                    shapes.append(input.shape[1:])
                else:
                    shapes.append(tuple(input.get_output_dim()[1:]))
            for i, shape in enumerate(shapes):
                prod = 1
                for dim in tuple(shape):
                    prod *= dim
                shapes[i] = prod
            self.input_shape = (1, sum(shapes), 1, 1)
        elif len(self.inputs) is 1:
            self.input_shape = self.inputs[0].shape
        else:
            raise Exception('Invalid merge type')
        output_dim = self.input_shape
        for layer in self.layers:
            output_dim = layer.get_output_dim(output_dim)
        return output_dim

    def set_final_output_shape(self, output_shape):
        for layer in reversed(self.layers):
            if layer.mods_io_dim:
                layer.set_final_output_shape(output_shape)
                return

    def get_num_data_inputs(self):
        num_data_inputs = 0
        inputs = self.inputs[:]
        for i in range(len(inputs)):
            if isinstance(inputs[i], np.ndarray):
                num_data_inputs += 1
            else:
                num_data_inputs += inputs[i].get_num_data_inputs()
        return num_data_inputs

    def get_output(self, params_list, data_tensors, testing=False):
        branch_params = params_list[0]
        param_i = 1
        updates = []
        for layer in self.layers:
            if layer.updates is not None:
                updates += layer.updates
        inputs = self.inputs[:]
        for i in range(len(inputs)):
            if isinstance(inputs[i], np.ndarray):
                inputs[i] = data_tensors.pop(0)
            else:
                inputs[i], us = inputs[i].get_output(params_list[param_i:], data_tensors, testing)
                param_i += 1
                updates += us
            if self.merge_mode is 'flat_append' and len(inputs) > 1:
                inputs[i] = inputs[i].flatten(2)
            elif self.merge_mode is not 'flat_append':
                raise Exception('Invalid merge mode')
        if len(inputs) > 1 and self.merge_mode is 'flat_append':
            x = T.concatenate(inputs, axis=1)
        else:
            x = inputs[0]
        current_layer = x
        for layer, params in zip(self.layers, branch_params):
            current_layer = layer.get_output(current_layer, params, testing)
        output = current_layer.astype('float32')
        return output, updates


class Model():
    def __init__(self):
        self.compiled = False
        self.dream_compiled = False
        self.set_loss(Losses.Crossentropy())
        self.set_accuracy(Accuracies.Categorical())
        self.set_optimizer(Optimizers.RMSprop())
        self.set_dream_optimizer(Optimizers.RMSprop(learning_rate=0.1, momentum=0.99))
        self.outputs = []

    def save_params(self, filename='okapi_params.pk'):
        file = open(filename, 'wb')
        pickle.dump(self.params_shared, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load_params(self, filename='okapi_params.pk'):
        file = open(filename, 'rb')
        self.params_shared = pickle.load(file)
        file.close()

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_dream_optimizer(self, optimizer):
        self.dream_optimizer = optimizer

    def set_loss(self, loss):
        self.loss = loss

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def add_output(self, output):
        self.outputs.append(output)

    def get_init_params(self):
        init_params = self.tree.get_init_params([])
        return init_params

    def initialize_params(self):
        self.init_params = self.get_init_params()
        self.params_shared = []
        for branch_params in self.init_params:
            branch_params_shared = []
            for layer_params in branch_params:
                if layer_params is not None:
                    layer_params_shared = []
                    for params in layer_params:
                        layer_params_shared.append(theano.shared(params))
                else:
                    layer_params_shared = None
                branch_params_shared.append(layer_params_shared)
            self.params_shared.append(branch_params_shared)

    def randomize_params(self):
        rand_params = self.get_init_params()
        for current_params, new_params in zip(self.params_shared, rand_params):
            if current_params is not None:
                for current, rand in zip(current_params, new_params):
                    current.set_value(rand)

    def set_params_as_vec(self, params):
        index = 0
        params = params.astype('float32')
        for branch_params in self.params_shared:
            for layer_params in branch_params:
                if layer_params is not None:
                    for shared in layer_params:
                        shape = shared.get_value().shape
                        size = np.prod(shape)
                        new_params = params[index:index + size].reshape(shape)
                        shared.set_value(new_params)
                        index += size

    def get_params_as_vec(self):
        vec = []
        for branch_params in self.params_shared:
            for layer_params in branch_params:
                if layer_params is not None:
                    for params in layer_params:
                        vec = np.append(vec, params.get_value().flat)
        return vec

    def set_tree(self, tree):
        self.tree = tree

    def compile(self, x_train, y_train, initialize_params=True):
        print('Compiling model...')
        self.compiled = True
        try:
            self.num_output_dims
        except:
            self.num_output_dims = y_train.ndim
        y_train = atleast_4d(y_train)
        for i in range(len(x_train)):
            x_train[i] = atleast_4d(x_train[i])
        self.tree.set_final_output_shape(y_train.shape)
        if initialize_params:
            self.initialize_params()

        y = T.tensor4(dtype='float32')
        num_data_inputs = self.tree.get_num_data_inputs()
        data_inputs = []
        for i in range(num_data_inputs):
            data_inputs.append(T.tensor4(dtype='float32'))

        y_hat, layer_updates = self.tree.get_output(self.params_shared, data_inputs[:], False)
        y_hat_test, layer_updates = self.tree.get_output(self.params_shared, data_inputs[:], True)

        all_params = []
        for branch_params in self.params_shared:
            for layer_params in branch_params:
                if layer_params is not None:
                    for params in layer_params:
                        all_params.append(params)

        train_loss = self.loss.get_train_loss(y_hat, y, all_params)
        test_loss = self.loss.get_test_loss(y_hat_test, y, all_params)
        test_acc = self.accuracy.get_accuracy(y_hat_test, y)

        preds = y_hat_test.flatten(self.num_output_dims)

        self.optimizer.build(all_params)
        updates = list(self.optimizer.get_updates(all_params, train_loss))
        for i, update in enumerate(updates):
            updates[i] = (update[0], update[1].astype('float32'))
        updates += layer_updates

        all_inputs = data_inputs + [y]

        self.train_loss_theano = theano.function(all_inputs, train_loss)
        self.test_loss_theano = theano.function(all_inputs, test_loss)
        self.test_acc_theano = theano.function(all_inputs, test_acc)
        self.predict_theano = theano.function(data_inputs, preds)
        self.update_step = theano.function(
            inputs=all_inputs,
            outputs=train_loss,
            updates=updates)

    def predict(self, x):
        for i in range(len(x)):
            x[i] = atleast_4d(x[i])
        preds_theano = self.predict_theano(*x)
        preds = []
        for dim in [output.shape for output in self.outputs]:
            preds.append(preds_theano[:sum(dim)])
        return preds

    def get_train_loss(self, x, y):
        for i in range(len(x)):
            x[i] = atleast_4d(x[i])
        y = atleast_4d(y)
        return self.train_loss_theano(*x, y)

    def get_test_loss(self, x, y):
        for i in range(len(x)):
            x[i] = atleast_4d(x[i])
        y = atleast_4d(y)
        return self.test_loss_theano(*x, y)

    def get_accuracy(self, x, y, batch_size=128, shuffle=True):
        for i in range(len(x)):
            x[i] = atleast_4d(x[i])
        y = atleast_4d(y)
        x_batches, y_batches, num_batches = make_batches(
            x, y, batch_size, shuffle=shuffle)
        accuracy = 0
        for x_batch, y_batch in zip(x_batches, y_batches):
            accuracy += self.test_acc_theano(*x_batch, y_batch)
        return accuracy / num_batches * 100

    def get_dream_accuracy(self, x, y, max_dream_length=24,
            initializer=Initializers.zeros):
        y = atleast_4d(y)
        preds = self.predict_dream(x, [y.shape[1:]], max_dream_length, initializer)
        accuracy = self.dream_accuracy_theano(preds[0].astype('float32'), y)
        return accuracy * 100, preds

    def write_progress(self, epoch, num_epochs, batch_num, num_batches,
                       time, loss):
        progress = ("\rEpoch {}/{} | Batch {}/{} | Time: {}s | Loss: {}   "
                    .format(epoch + 1, num_epochs,
                            batch_num + 1, num_batches,
                            round(time, 1),
                            loss))
        sys.stdout.write(progress)

    def est_time_remaining(self, last_time, iteration, num_iterations):
        iterations_left = num_iterations - iteration - 1
        return last_time * iterations_left

    def compile_dream(self, x_train, shapes, indices, initializer):
        self.dream_compiled = True
        self.x_dream = []
        index = 0
        for i in range(len(x_train)):
            if i in indices:
                self.x_dream.append(theano.shared(initializer(shapes[index]).astype('float32')))
                index += 1
            else:
                x_train[i] = atleast_4d(x_train[i][[0]])
                self.x_dream.append(theano.shared(x_train[i].astype('float32')))

        y_hat_test, layer_updates = self.tree.get_output(self.params_shared, self.x_dream[:], True)
        preds = y_hat_test.flatten(self.num_output_dims).mean(axis=None)

        self.dream_optimizer.build([self.x_dream[index] for index in indices])
        updates = list(self.dream_optimizer.get_updates([self.x_dream[index] for index in indices], -preds))
        for i, update in enumerate(updates):
            updates[i] = (update[0], update[1].astype('float32'))
        updates += layer_updates

        y_pred = T.tensor4(dtype='float32')
        y = T.tensor4(dtype='float32')
        accuracy = self.accuracy.get_accuracy(y_pred, y)

        self.dream_accuracy_theano = theano.function([y_pred, y], accuracy)
        self.dream_update = theano.function(
            inputs=[],
            outputs=preds,
            updates=updates
        )

    def predict_dream(self, x_train, shapes, max_dream_length=24,
            initializer=Initializers.zeros):
        for i, shape in enumerate(shapes):
            shape = list(shape)
            shape = [1] + shape
            shape = atleast_4d(np.zeros(tuple(shape))).shape
            shapes[i] = shape
        indices = []
        for i in range(len(x_train)):
            if x_train[i] is None:
                indices.append(i)
        if not self.dream_compiled:
            self.compile_dream(x_train[:], shapes, indices, initializer)
        for i in range(len(x_train)):
            if i not in indices:
                x_train[i] = atleast_4d(x_train[i])
                num_rows = x_train[i].shape[0]
        predictions = []
        for shape in shapes:
            prediction_shape = list(shape)
            prediction_shape[0] = num_rows
            predictions.append(np.zeros(prediction_shape))
        for row in range(num_rows):
            index = 0
            for i in range(len(x_train)):
                if i in indices:
                    self.x_dream[i].set_value(
                            atleast_4d(initializer(shapes[index])))
                    index += 1
                else:
                    self.x_dream[i].set_value(atleast_4d(x_train[i][[row]]))
            for i in range(max_dream_length):
                reward = self.dream_update()
            if (row + 1) % 1000 is 0:
                print('{}/{}: {}'.format(row + 1, x_train[0].shape[0], reward))
            for pred_ind, index in enumerate(indices):
                predictions[pred_ind][row, :, :, :] = self.x_dream[index].get_value()
        return predictions

    def train(self, x, y, num_epochs=12, shuffle=True,
              params_filename='okapi_params.pk',
              initialize_params=True,
              batch_size=128):
        self.num_output_dims = y.ndim
        for i in range(len(x)):
            x[i] = atleast_4d(x[i])
        y = atleast_4d(y)
        if not self.compiled:
            self.compile(x, y, initialize_params=initialize_params)
        x_batches, y_batches, num_batches = make_batches(
            x, y, batch_size, shuffle)
        print('Started training...')
        for epoch in range(num_epochs):
            epoch_start = time.clock()
            if shuffle:
                x_batches, y_batches, num_batches = make_batches(
                    x, y, batch_size, shuffle=True)
            total_loss = 0
            for x_batch, y_batch, batch_num in zip(
                    x_batches, y_batches, range(num_batches)):
                batch_start = time.clock()
                loss = self.update_step(*x_batch, y_batch)
                total_loss += loss
                batch_time = time.clock() - batch_start
                time_rem = self.est_time_remaining(
                    batch_time, batch_num, num_batches)
                self.write_progress(epoch, num_epochs,
                                    batch_num, num_batches,
                                    time_rem, loss)

            epoch_time = time.clock() - epoch_start
            avg_loss = total_loss / num_batches
            self.write_progress(epoch, num_epochs,
                                num_batches - 1, num_batches,
                                epoch_time, avg_loss)
            print()
