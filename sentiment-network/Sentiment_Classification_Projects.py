import pdb

# coding: utf-8

# # Sentiment Classification & How To "Frame Problems" for a Neural Network
# 
# by Andrew Trask
# 
# - **Twitter**: @iamtrask
# - **Blog**: http://iamtrask.github.io

# ### What You Should Already Know
# 
# - neural networks, forward and back-propagation
# - stochastic gradient descent
# - mean squared error
# - and train/test splits
# 
# ### Where to Get Help if You Need it
# - Re-watch previous Udacity Lectures
# - Leverage the recommended Course Reading Material - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) (Check inside your classroom for a discount code)
# - Shoot me a tweet @iamtrask
# 
# 
# ### Tutorial Outline:
# 
# - Intro: The Importance of "Framing a Problem" (this lesson)
# 
# - [Curate a Dataset](#lesson_1)
# - [Developing a "Predictive Theory"](#lesson_2)
# - [**PROJECT 1**: Quick Theory Validation](#project_1)
# 
# 
# - [Transforming Text to Numbers](#lesson_3)
# - [**PROJECT 2**: Creating the Input/Output Data](#project_2)
# 
# 
# - Putting it all together in a Neural Network (video only - nothing in notebook)
# - [**PROJECT 3**: Building our Neural Network](#project_3)
# 
# 
# - [Understanding Neural Noise](#lesson_4)
# - [**PROJECT 4**: Making Learning Faster by Reducing Noise](#project_4)
# 
# 
# - [Analyzing Inefficiencies in our Network](#lesson_5)
# - [**PROJECT 5**: Making our Network Train and Run Faster](#project_5)
# 
# 
# - [Further Noise Reduction](#lesson_6)
# - [**PROJECT 6**: Reducing Noise by Strategically Reducing the Vocabulary](#project_6)
# 
# 
# - [Analysis: What's going on in the weights?](#lesson_7)

# # Lesson: Curate a Dataset<a id='lesson_1'></a>
# The cells from here until Project 1 include code Andrew shows in the videos leading up to mini project 1. We've included them so you can run the code along with the videos without having to type in everything.

# In[ ]:


def pretty_print_review_and_label(i):
    print(labels[i] + "\t:\t" + reviews[i][:80] + "...")

g = open('reviews.txt','r') # What we know!
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r') # What we WANT to know!
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()


# **Note:** The data in `reviews.txt` we're using has already been preprocessed a bit and contains only lower case characters. If we were working from raw data, where we didn't know it was all lower case, we would want to add a step here to convert it. That's so we treat different variations of the same word, like `The`, `the`, and `THE`, all the same way.

# In[ ]:


len(reviews)


# In[ ]:


reviews[0]


# In[ ]:


labels[0]


# # Lesson: Develop a Predictive Theory<a id='lesson_2'></a>

# In[ ]:


print("labels.txt \t : \t reviews.txt\n")
pretty_print_review_and_label(2137)
pretty_print_review_and_label(12816)
pretty_print_review_and_label(6267)
pretty_print_review_and_label(21934)
pretty_print_review_and_label(5297)
pretty_print_review_and_label(4998)


# # Project 1: Quick Theory Validation<a id='project_1'></a>
# 
# There are multiple ways to implement these projects, but in order to get your code closer to what Andrew shows in his solutions, we've provided some hints and starter code throughout this notebook.
# 
# You'll find the [Counter](https://docs.python.org/2/library/collections.html#collections.Counter) class to be useful in this exercise, as well as the [numpy](https://docs.scipy.org/doc/numpy/reference/) library.

# In[ ]:


from collections import Counter
import numpy as np


# We'll create three `Counter` objects, one for words from postive reviews, one for words from negative reviews, and one for all the words.

# In[ ]:


# Create three Counter objects to store positive, negative and total counts
positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()


# **TODO:** Examine all the reviews. For each word in a positive review, increase the count for that word in both your positive counter and the total words counter; likewise, for each word in a negative review, increase the count for that word in both your negative counter and the total words counter.
# 
# **Note:** Throughout these projects, you should use `split(' ')` to divide a piece of text (such as a review) into individual words. If you use `split()` instead, you'll get slightly different results than what the videos and solutions show.

# In[ ]:


# TODO: Loop over all the words in all the reviews and increment the counts in the appropriate counter objects
for i, review in enumerate(reviews):
    words = review.split(' ')
    if labels[i] == "POSITIVE":
        positive_counts.update(words)
    else:
        negative_counts.update(words)
    total_counts.update(words)


# Run the following two cells to list the words used in positive reviews and negative reviews, respectively, ordered from most to least commonly used. 

# In[ ]:


# Examine the counts of the most common words in positive reviews
positive_counts.most_common()


# In[ ]:


# Examine the counts of the most common words in negative reviews
negative_counts.most_common()


# As you can see, common words like "the" appear very often in both positive and negative reviews. Instead of finding the most common words in positive or negative reviews, what you really want are the words found in positive reviews more often than in negative reviews, and vice versa. To accomplish this, you'll need to calculate the **ratios** of word usage between positive and negative reviews.
# 
# **TODO:** Check all the words you've seen and calculate the ratio of postive to negative uses and store that ratio in `pos_neg_ratios`. 
# >Hint: the positive-to-negative ratio for a given word can be calculated with `positive_counts[word] / float(negative_counts[word]+1)`. Notice the `+1` in the denominator – that ensures we don't divide by zero for words that are only seen in positive reviews.

# In[ ]:


# Create Counter object to store positive/negative ratios
pos_neg_ratios = Counter()

# TODO: Calculate the ratios of positive and negative uses of the most common words
#       Consider words to be "common" if they've been used at least 100 times
for word_count in total_counts.most_common():
    w,c = word_count
    if c >= 100:
        pos_neg_ratios[w] = positive_counts[w] / (negative_counts[w] + 1.0)
    else:
        break


# Examine the ratios you've calculated for a few words:

# In[ ]:


print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))


# Looking closely at the values you just calculated, we see the following:
# 
# * Words that you would expect to see more often in positive reviews – like "amazing" – have a ratio greater than 1. The more skewed a word is toward postive, the farther from 1 its positive-to-negative ratio  will be.
# * Words that you would expect to see more often in negative reviews – like "terrible" – have positive values that are less than 1. The more skewed a word is toward negative, the closer to zero its positive-to-negative ratio will be.
# * Neutral words, which don't really convey any sentiment because you would expect to see them in all sorts of reviews – like "the" – have values very close to 1. A perfectly neutral word – one that was used in exactly the same number of positive reviews as negative reviews – would be almost exactly 1. The `+1` we suggested you add to the denominator slightly biases words toward negative, but it won't matter because it will be a tiny bias and later we'll be ignoring words that are too close to neutral anyway.
# 
# Ok, the ratios tell us which words are used more often in postive or negative reviews, but the specific values we've calculated are a bit difficult to work with. A very positive word like "amazing" has a value above 4, whereas a very negative word like "terrible" has a value around 0.18. Those values aren't easy to compare for a couple of reasons:
# 
# * Right now, 1 is considered neutral, but the absolute value of the postive-to-negative rations of very postive words is larger than the absolute value of the ratios for the very negative words. So there is no way to directly compare two numbers and see if one word conveys the same magnitude of positive sentiment as another word conveys negative sentiment. So we should center all the values around netural so the absolute value fro neutral of the postive-to-negative ratio for a word would indicate how much sentiment (positive or negative) that word conveys.
# * When comparing absolute values it's easier to do that around zero than one. 
# 
# To fix these issues, we'll convert all of our ratios to new values using logarithms.
# 
# **TODO:** Go through all the ratios you calculated and convert them to logarithms. (i.e. use `np.log(ratio)`)
# 
# In the end, extremely positive and extremely negative words will have positive-to-negative ratios with similar magnitudes but opposite signs.

# In[ ]:


# TODO: Convert ratios to logs
for w,c in pos_neg_ratios.most_common():
    pos_neg_ratios[w] = np.log(c)


# Examine the new ratios you've calculated for the same words from before:

# In[ ]:


print("Pos-to-neg ratio for 'the' = {}".format(pos_neg_ratios["the"]))
print("Pos-to-neg ratio for 'amazing' = {}".format(pos_neg_ratios["amazing"]))
print("Pos-to-neg ratio for 'terrible' = {}".format(pos_neg_ratios["terrible"]))


# If everything worked, now you should see neutral words with values close to zero. In this case, "the" is near zero but slightly positive, so it was probably used in more positive reviews than negative reviews. But look at "amazing"'s ratio - it's above `1`, showing it is clearly a word with positive sentiment. And "terrible" has a similar score, but in the opposite direction, so it's below `-1`. It's now clear that both of these words are associated with specific, opposing sentiments.
# 
# Now run the following cells to see more ratios. 
# 
# The first cell displays all the words, ordered by how associated they are with postive reviews. (Your notebook will most likely truncate the output so you won't actually see *all* the words in the list.)
# 
# The second cell displays the 30 words most associated with negative reviews by reversing the order of the first list and then looking at the first 30 words. (If you want the second cell to display all the words, ordered by how associated they are with negative reviews, you could just write `reversed(pos_neg_ratios.most_common())`.)
# 
# You should continue to see values similar to the earlier ones we checked – neutral words will be close to `0`, words will get more positive as their ratios approach and go above `1`, and words will get more negative as their ratios approach and go below `-1`. That's why we decided to use the logs instead of the raw ratios.

# In[ ]:


# words most frequently seen in a review with a "POSITIVE" label
pos_neg_ratios.most_common()


# In[ ]:


# words most frequently seen in a review with a "NEGATIVE" label
list(reversed(pos_neg_ratios.most_common()))[0:30]

# Note: Above is the code Andrew uses in his solution video, 
#       so we've included it here to avoid confusion.
#       If you explore the documentation for the Counter class, 
#       you will see you could also find the 30 least common
#       words like this: pos_neg_ratios.most_common()[:-31:-1]


# # End of Project 1. 
# ## Watch the next video to see Andrew's solution, then continue on to the next lesson.
# 
# # Transforming Text into Numbers<a id='lesson_3'></a>
# The cells here include code Andrew shows in the next video. We've included it so you can run the code along with the video without having to type in everything.

# In[ ]:


from IPython.display import Image

review = "This was a horrible, terrible movie."

Image(filename='sentiment_network.png')


# In[ ]:


review = "The movie was excellent"

Image(filename='sentiment_network_pos.png')


# # Project 2: Creating the Input/Output Data<a id='project_2'></a>
# 
# **TODO:** Create a [set](https://docs.python.org/3/tutorial/datastructures.html#sets) named `vocab` that contains every word in the vocabulary.

# In[ ]:


# TODO: Create set named "vocab" containing all of the words from all of the reviews
vocab = set(total_counts)


# Run the following cell to check your vocabulary size. If everything worked correctly, it should print **74074**

# In[ ]:


vocab_size = len(vocab)
print(vocab_size)


# Take a look at the following image. It represents the layers of the neural network you'll be building throughout this notebook. `layer_0` is the input layer, `layer_1` is a hidden layer, and `layer_2` is the output layer.

# In[ ]:


from IPython.display import Image
Image(filename='sentiment_network_2.png')


# **TODO:** Create a numpy array called `layer_0` and initialize it to all zeros. You will find the [zeros](https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html) function particularly helpful here. Be sure you create `layer_0` as a 2-dimensional matrix with 1 row and `vocab_size` columns. 

# In[ ]:


# TODO: Create layer_0 matrix with dimensions 1 by vocab_size, initially filled with zeros
layer_0 = np.zeros((1,len(vocab)))


# Run the following cell. It should display `(1, 74074)`

# In[ ]:


layer_0.shape


# In[ ]:


from IPython.display import Image
Image(filename='sentiment_network.png')


# `layer_0` contains one entry for every word in the vocabulary, as shown in the above image. We need to make sure we know the index of each word, so run the following cell to create a lookup table that stores the index of every word.

# In[ ]:


# Create a dictionary of words in the vocabulary mapped to index positions
# (to be used in layer_0)
word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i
    
# display the map of words to indices
word2index


# **TODO:**  Complete the implementation of `update_input_layer`. It should count 
#           how many times each word is used in the given review, and then store
#           those counts at the appropriate indices inside `layer_0`.

# In[ ]:


def update_input_layer(review):
    """ Modify the global layer_0 to represent the vector form of review.
    The element at a given index of layer_0 should represent
    how many times the given word occurs in the review.
    Args:
        review(string) - the string of the review
    Returns:
        None
    """
    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0
    
    # TODO: count how many times each word is used in the given review and store the results in layer_0
    c = Counter(review.split(' '))
    for w, n in c.items():
        layer_0[0, word2index[w]] = n


# Run the following cell to test updating the input layer with the first review. The indices assigned may not be the same as in the solution, but hopefully you'll see some non-zero values in `layer_0`.  

# In[ ]:


update_input_layer(reviews[0])
layer_0


# **TODO:** Complete the implementation of `get_target_for_labels`. It should return `0` or `1`, 
#           depending on whether the given label is `NEGATIVE` or `POSITIVE`, respectively.

# In[ ]:


def get_target_for_label(label):
    """Convert a label to `0` or `1`.
    Args:
        label(string) - Either "POSITIVE" or "NEGATIVE".
    Returns:
        `0` or `1`.
    """
    # TODO: Your code here
    r = 1 if label == "POSITIVE" else 0
    return r


# Run the following two cells. They should print out`'POSITIVE'` and `1`, respectively.

# In[ ]:


labels[0]


# In[ ]:


get_target_for_label(labels[0])


# Run the following two cells. They should print out `'NEGATIVE'` and `0`, respectively.

# In[ ]:


labels[1]


# In[ ]:


get_target_for_label(labels[1])


# # End of Project 2. 
# ## Watch the next video to see Andrew's solution, then continue on to the next lesson.

# # Project 3: Building a Neural Network<a id='project_3'></a>

# **TODO:** We've included the framework of a class called `SentimentNetork`. Implement all of the items marked `TODO` in the code. These include doing the following:
# - Create a basic neural network much like the networks you've seen in earlier lessons and in Project 1, with an input layer, a hidden layer, and an output layer. 
# - Do **not** add a non-linearity in the hidden layer. That is, do not use an activation function when calculating the hidden layer outputs.
# - Re-use the code from earlier in this notebook to create the training data (see `TODO`s in the code)
# - Implement the `pre_process_data` function to create the vocabulary for our training data generating functions
# - Ensure `train` trains over the entire corpus

# ### Where to Get Help if You Need it
# - Re-watch earlier Udacity lectures
# - Chapters 3-5 - [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning) - (Check inside your classroom for a discount code)

# In[ ]:


import time
import sys
import numpy as np

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training
        
        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development 
        np.random.seed(1)

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)
        
        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):
        
        # populate review_vocab with all of the words in the given reviews
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)
        
        # populate label_vocab with all of the words in the given labels.
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)
        
        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)
        
        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i
        
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

        #pdb.set_trace()
        
    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # These are the weights between the input layer and the hidden layer.
        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))
    
        # These are the weights between the hidden layer and the output layer.
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, 
                                                (self.hidden_nodes, self.output_nodes))
        
        # The input layer, a two-dimensional matrix with shape 1 x input_nodes
        self.layer_0 = np.zeros((1,input_nodes))
    

    def update_input_layer(self,review):

        # clear out previous state, reset the layer to be all 0s
        self.layer_0 *= 0

        for word in review.split(" "):
          if(word in self.word2index.keys()):
              self.layer_0[0][self.word2index[word]] += 1
        
        #c = Counter(review.split(' '))
        #for w, n in c.items():
        #    if w in self.word2index:
        #        self.layer_0[0][self.word2index[w]] = n
                
    def get_target_for_label(self,label):
        # TODO: Copy the code you wrote for get_target_for_label 
        #       earlier in this notebook. 
        #return 1 if label == "POSITIVE" else 0
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0
        
    def sigmoid(self,x):
        # TODO: Return the result of calculating the sigmoid activation function
        #       shown in the lectures
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_output_2_derivative(self,output):
        # TODO: Return the derivative of the sigmoid activation function, 
        #       where "output" is the original output from the sigmoid fucntion 
        return output * (1 - output)

    def train(self, training_reviews, training_labels):
        
        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))
        
        # Keep track of correct predictions to display accuracy during training 
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):
            
            # TODO: Get the next review and its correct label
            training_review = training_reviews[i]
            training_label = training_labels[i]
            
            # TODO: Implement the forward pass through the network. 
            #       That means use the given review to update the input layer, 
            #       then calculate values for the hidden layer,
            #       and finally calculate the output layer.
            # 
            #       Do not use an activation function for the hidden layer,
            #       but use the sigmoid activation function for the output layer.
            self.update_input_layer(training_review)
            
            # Hidden layer
            layer_1 = self.layer_0.dot(self.weights_0_1)

            # Output layer
            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
            
            # TODO: Implement the back propagation pass here. 
            #       That means calculate the error for the forward pass's prediction
            #       and update the weights in the network according to their
            #       contributions toward the error, as calculated via the
            #       gradient descent and back propagation algorithms you 
            #       learned in class.
            layer_2_error = layer_2 - self.get_target_for_label(training_label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error
            
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate
            
            # TODO: Keep track of correct predictions. To determine if the prediction was
            #       correct, check that the absolute value of the output error 
            #       is less than 0.5. If so, add one to the correct_so_far count.
            if(layer_2 >= 0.5 and training_label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and training_label == 'NEGATIVE'):
                correct_so_far += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the training process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1)                              + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")
    
    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """
        
        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label. 
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1
            
            # For debug purposes, print out our prediction accuracy and speed 
            # throughout the prediction process. 

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            
            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4]                              + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5]                              + " #Correct:" + str(correct) + " #Tested:" + str(i+1)                              + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")
    
    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # TODO: Run a forward pass through the network, like you did in the
        #       "train" function. That means use the given review to 
        #       update the input layer, then calculate values for the hidden layer,
        #       and finally calculate the output layer.
        #
        #       Note: The review passed into this function for prediction 
        #             might come from anywhere, so you should convert it 
        #             to lower case prior to using it.
        self.update_input_layer(review.lower())
        layer_1 = self.layer_0.dot(self.weights_0_1)
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        if (layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"


# Run the following cell to create a `SentimentNetwork` that will train on all but the last 1000 reviews (we're saving those for testing). Here we use a learning rate of `0.1`.

# In[ ]:


#mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.1)


# Run the following cell to test the network's performance against the last 1000 reviews (the ones we held out from our training set). 
# 
# **We have not trained the model yet, so the results should be about 50% as it will just be guessing and there are only two possible values to choose from.**

# In[ ]:


#mlp.test(reviews[-1000:],labels[-1000:])


# Run the following cell to actually train the network. During training, it will display the model's accuracy repeatedly as it trains so you can see how well it's doing.

# In[ ]:


#mlp.train(reviews[:-1000],labels[:-1000])


# That most likely didn't train very well. Part of the reason may be because the learning rate is too high. Run the following cell to recreate the network with a smaller learning rate, `0.01`, and then train the new network.

# In[ ]:


#mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.01)
#mlp.train(reviews[:-1000],labels[:-1000])


# That probably wasn't much different. Run the following cell to recreate the network one more time with an even smaller learning rate, `0.001`, and then train the new network.

# In[ ]:


mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
mlp.train(reviews[:-1000],labels[:-1000])


# With a learning rate of `0.001`, the network should finall have started to improve during training. It's still not very good, but it shows that this solution has potential. We will improve it in the next lesson.

# # End of Project 3. 
