# Simpson-o-matic #
###### A modest attempt at generating a Simpsons script using an RNN #####

#### Introduction
As a winter project, I'm attempting to use a Recursive Neural Network to
generate an episode script from the long-running animated show _The Simpsons_.
At the time of this writing, there have been 627 episodes of _The Simpsons_,
which provides a substantial corpus of data to be used in training the RNN.

The project will be using the **TensorFlow** library to create and train the RNN,
the Python package **BeautifulSoup** to perform web-scraping for retrieving the
scripts, and, as a stretch goal, _intends_ to use **Go** to efficiently tokenize 
the scripts into feature vectors to be used by **TensorFlow**.

#### Languages and Frameworks

Python 3.6.3

Tensorflow 1.4.0

beautifulsoup4 - 4.6.0


#### History

12/16/17 - Initial Commit