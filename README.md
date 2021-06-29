Federated and Distributed Learning

This project deals with the aspect of federated learning and distributed computations.
The goal of the project is to have a modular set up where different nodes in a network have local batches of data samples.
All of the models instantiated on the workers are exactly identical but might have different intializations.
Hence, the training pipeline goes by performing a forward pass over each batch at each of the workers, followed by a gradient computation.
At the server level, all workers send their gradients and the server performs an average for all gradients and sends a copy to each of the workers.
At the final stage each worker's gradient is updated with the average gradient of all workers and an optimization step is done.
With that, instead of maximizing training for only a local instance of a given dataset we are trying to maximize the performance over different instances of the same dataset.
Nevertheless, such architecture is able to account for infinitely large dataset where each worker node can have limited computation power.

The Project has been implemented using PyTorch.