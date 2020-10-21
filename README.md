# K-Nearest-Neighbors
This is a simple Nearest Neighbors algorithm built on regular python and a little bit of NumPy.

This algorithm is used for classification problems. It is a simple, brute-force model so 
it will take a while to run for larger data sets. Compared to sci-kit learn's nearest neighbors classifier, it does very well. Tested on two small data sets and the output for all of the test data was exactly the same.

To use this algorithm, instatiate the class with the number of neighbors you want to use to predict the output.

Such as: '''neigh = KNearestNeighbors(n_neighbors=5)'''

No need to encode the class labels, the model will do that for.
It will return the class label names in the prediction as well.

