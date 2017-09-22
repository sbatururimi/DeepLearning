# The function decoding_layer is implemented correctly.

For the f.random_uniform, I would suggest to add a range of value for the initialization, as -1 to 1

# A range of value that is good for this project:

epochs = [5-15] : More you have layers, more you should increase the number of epochs, the key is to choose a number such that the loss on validation set stops decreasing further.
batch_size = [256, 1024] : mainly depend on where you will run the code, and it's power
rnn_size = [128, 512] : High mean, the model will learn more complex structure
num_layers = [2, 4] : The more you have layers, the more complex the model will be
embedding_size = [128, 256] : Can represent the number of unique words we can deal with.
learning_rate = [0.001, 0.01] : low to learn form the large variability of the dataset
keep_probability = [0.6, 0.9] : Depend where you put dropout layer, but shouldn't be too low due to the small size of our dataset
Don't forget: the values of hyper parameters should be power of 2. Tensorflow optimizes our computation if we do so.

# It would be even better with a graph:

# Visualize the loss and accuracy
import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
ax1.plot(loss_list, color='red')
ax1.set_title('Traning Loss')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss value')
ax2.plot(valid_acc_list)
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy')
plt.show()


# The function sentence_to_seq is implemented correctly.

Good !
Here another possible implementation
return [vocab_to_int.get(w, vocab_to_int['<UNK>']) for w in sentence.lower().split()]