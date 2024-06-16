
# https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#example-logistic-regression-bag-of-words-classifier

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# find out if a sentence is SPANISH or ENGLISH

# data representation
# bag of words vector, each word is assigned an index and bow vector is reresented by frequencies of word
# for sentence "hello world hello", bow=[2,1]  if 0 index is for hello and 1 index is for world

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]


test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

# size of dictionary we have from training data and test data, the words we know
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2 # we have two labels SPANISH AND ENGLISH

class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # calls the init function of nn.Module.  Dont get confused by syntax,
        # just always do it in an nn.Module
        super(BoWClassifier, self).__init__()

        # Define the parameters that you will need.  In this case, we need A and b,
        # the parameters of the affine mapping.
        # Torch defines nn.Linear(), which provides the affine map.
        # Make sure you understand why the input dimension is vocab_size
        # and the output is num_labels!
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec):
        # Pass the input through the linear layer,
        # then pass that through log_softmax.
        # Many non-linearities and other functions are in torch.nn.functional
        return F.log_softmax(self.linear(bow_vec), dim=1)

    ### custom function for predicting label
    def predict(self, instance):
        bow_vec = make_bow_vector(instance, word_to_ix)
        log_probs = self(bow_vec)
        return "SPANISH" if log_probs[0][0] > log_probs[0][1] else "ENGLISH"
        
    def train_once(self, data, loss_function, optimizer):
        for instance, label in data:
        # Step 1. Remember that PyTorch accumulates gradients.
        # We need to clear them out before each instance
          self.zero_grad()

        # Step 2. Make our BOW vector and also we must wrap the target in a
        # Tensor as an integer. For example, if the target is SPANISH, then
        # we wrap the integer 0. The loss function then knows that the 0th
        # element of the log probabilities is the log probability
        # corresponding to SPANISH
          bow_vec = make_bow_vector(instance, word_to_ix)
          target = make_target(label, label_to_ix)
          print("target: "+label)
          print(target)

        # Step 3. Run our forward pass.
          log_probs = self(bow_vec)
          print("log_probs: "+label)
          print(log_probs)
          
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
          loss = loss_function(log_probs, target)
          loss.backward()
          optimizer.step()


## convert sentence to vector space
def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

## target is the target label
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)


# the model knows its parameters.  The first output below is A, the second is b.
# Whenever you assign a component to a class variable in the __init__ function
# of a module, which was done with the line
# self.linear = nn.Linear(...)
# Then through some Python magic from the PyTorch devs, your module
# (in this case, BoWClassifier) will store knowledge of the nn.Linear's parameters
for param in model.parameters():
    print(param)


# To run the model, pass in a BoW vector
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    sample = data[0]
    print(model.predict(sample[0]))    




# before training
with torch.no_grad():
    for instance, label in test_data:
        print(model.predict(instance)) 


print(next(model.parameters())[:, word_to_ix["creo"]])

## define cost function
## negative log likelihood loss
loss_function = nn.NLLLoss()

## gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)


## repeat 100 time training
for epoch in range(100):
    model.train_once(data,loss_function,optimizer)


with torch.no_grad():
    for instance, label in test_data:
        print("predict for sentence : "+str(instance) +", actual label"+label)
        print(model.predict(instance))

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])        
