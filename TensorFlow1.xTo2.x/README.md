# TensorFlow 1.x To 2.x
Notes on the same below.

# TensorFlow 1.x computational graph program structure
In TensorFlow 1.x we divide our program into two separate parts: a definition of a computational graph and its execution. 
As we know, a computer graph has two parts, nodes and edges. In. our favour, all the data to be used which are tensor objects such as constants, variables and placeholders and all the computations to be performed that are operation objects are defined. 

Nodes in the netwpork represent objects (tensors and operations) and edges represent the tensors that flow between these operations. 

To execute such a graph, the session object is to be used. The values of different tensor objects are initialised, accessed and saved in a session object only. Until this point, the tensor objects were just abstract definitions. 

Now, we work up an example that can be found [here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/GraphExample_TensorFlow1_x.ipynb). 

# Constants, Variables and Placeholders

We all know what constants are. Variables here are also tensors that require updating within a session. A good example would be the weight values in a neural network. These variables need to be explicitly initialised before use. 

Constants are stored in a computational graph definition and are loaded every time the graph is loaded, so they are memory intensive.
Variables, on the other hand are stored separately, they can exist on parameter servers.

Placeholders are used to feed values in a TensorFlow graph. They are used along with feed_dict to feed data in `run(fetches, feed_dict=none, options=None, run_metadata)`.
Normally used to feed new training examples while training a neural network. As they do not have any data, they need not be initialised. They allow us to create our operations and build the computational graph without requiring any data. 

We also worked up an example for the same, [here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/ConstantVariablesSequencesRandom_Example.ipynb).

# Now, TensorFlow 1.x was a lower-level API

You build models by creating a graph of ops which you then compile.

`tf.keras` offers higher level APIs

# Sequential API

A very intuitive, elegant and consise model used much often. Not much discussion needed.

# Functional API

It is useful when you want to build a model with more complex or non-linear topologies that include multiple inputs, multiple outputs and residual connections with non-sequential flows, shared and reusable layers. 
Each layer is callable and each returns a tensor as an output.

An example is provided [here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/FunctionalAPI_example.ipynb).

# Model Subclassing

It offers the highest flexibility and is generally only used when you need to define your own layer. Higher complexity cost. We subclass `tf.keras.layers.Layer` and implement one of the following methods.

- `__innit__` is optionally used to define all the sublayers to be used by this layer. It is a constructor where we can define our model.
- `build` is used to create the weights of the layer. we use `add_weight()` to add weights. 
- `call` is used to define the forward pass. We call our layer here and chain it in functional style.
- A layer can also be serialised by using `get_config()` and deserialised using `from_config()`.

[Here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/ModelSubclassing_Example.ipynb) is an example of a custom layer that simply multiplies an input by a matrix named kernel.

# Callbacks

Callbacks are objects passed to a model to extend or modify behaviors during training. 

- `tf.keras.callbacks.ModelCheckpoint` is used to make checkpoints of our model at points to be able to recover incase of any problems.
- `tf.keras.callbacks.LearningRateScheduler` is used to dynamically change the learning rate during optimisation.
- `tf.keras.callbacks.EarlyStopping ` is used to interrupt training when validation performance has stopped improving after a while.
- `tf.keras.callbacks.TensorBoard` is used to monitor the behavior.

An example would be as follows, 

```
callbacks = [
  # Write TensorBoard logs to './logs' directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
 ]
 
 model.fit(data, labels, batch_size=256, epochs=100, callbacks=callbacks, validation_data=(val_data, val_labels))
 
 ```
 
# Model and the Weights

After training a model, it is useful to save those weights.

```
# Saving weights to a tensorflow checkpoint file
model.save_weights('./weights/my_model)

```
Weights in 2.x can also be easily saved in Keras format which is more portable across multiple backends.

```
# Saving weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

```
Weights can be easily loaded with,

```
# Restore the model's state
model.load_weights(file_path)

```
In addition to weights, the model can be serialised in JSON with,

```
json_string = model.to_json() # save
model = tf.keras.models.model_from_json(json_string) # restore

```
If we wish to save a model together with its weights and optimisation parameters, we use,

```
model.save('my_model.h5') # save
model = tf.keras.models.load_model('my_model.h5') # restore

```


# tf.keras or Estimators?

In additon to direct graph computation and to the higher level APIs tf.keras provides we also have Estimator support in both 1.x and 2.x.
Estimators are highly efficient learning models for large scale production-ready environments, which can be trained on single machines or on distibuted multi-servers, and they run on CPUs, GPUs or TPUs without recording your model.

[Here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/EstimatorsBasic_Example.ipynb), is an example to better understand. 
[Here](https://github.com/soumilhooda/TensorFlow1.xTo2.x/blob/main/MNIST_estimator.ipynb) is an example of MNIST classification using an estimator.


# Ragged Tensors

2.x brought in the support for ragged tensors as well. Ragged tensors are special tensors with non-uniformly shaped dimensions. This is particularly useful for dealing with sequences and other data issues where the dimension can change across batches, such as text sentences and hierarchical data.

This is more efficient than padding `tf.Tensor` as no time or space is wasted.

# Custom Training

If you use `tf.keras` then you will easily train your model with `fit()` and probably never need to go in detail of how the gradients are computed internally. If you want finer control over optimization you would want custom training.

- `tf.GradientTape()` is a class that records operations for automatic differentiation. 
  ```
  x = tf.constant(4.0)
  with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x*x
    z = y*y
   dz_dx = g.gradient(z,x) # 256.0 (4*x^3 at x=4)
   dy_dz = g.gradient(y,x) # 8.0
   print(dz_dx)
   print(dy_dx)
   del g # Drop the reference to the tape
- `tf.gradient_function()` returns a function that computes the derivatives of its input function paramter with respect to its parameters.
- `tf.value_and_gradients_function()` returns the value from the input function in addition to the list of derivatives of the input function with respect to its arguements.
- `tf.implicit_gradients()` computes the gradients of the outputs of the input function with regards to all trainable variablees these outputs depend on.


# Distributed Training in 2.x

Distributed GPUs, the advantage 2.x holds, multiple machines, and TPUs just at the cost of few additional lines. 
`tf.distibute.Strategy` is the API used and it supports both `tf.keras` and `tf.estimator` and also supports eager execution.
Stratigies can be synchronous, where all workers train over different slices of input data in a form of sync data parallel computation, or asynchornous, where updates from the optimizers are not happening in sync. 
All strategies require that the data is loaded in batches in `tf.data.Dataset` API.






