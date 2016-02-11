from OkapiV2.Core import Model, Branch
from OkapiV2.Layers.Basic import FullyConnected
from OkapiV2.Layers.Activations import ActivationLayer, PReLULayer
from OkapiV2 import Activations, Datasets, Optimizers

x_train, y_train, x_val, y_val, x_test, y_test = Datasets.load_mnist()

tree = Branch()
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected((512, 1, 1, 1)))
tree.add_layer(PReLULayer())
tree.add_layer(FullyConnected())
tree.add_layer(ActivationLayer(Activations.softmax))
tree.add_input(x_train)

model = Model()
model.set_tree(tree)
model.set_optimizer(Optimizers.RMSprop(learning_rate=0.00005))

index = 60000
model.train([x_train[:index, :, :, :]], y_train[:index, :], 24)
accuracy = model.get_accuracy([x_train[:index, :, :, :]], y_train[:index])
print('Accuracy: {}%'.format(accuracy))
test_accuracy = model.get_accuracy([x_test], y_test)
print('Test accuracy: {}%'.format(test_accuracy))
