# K-Nearest-Neighbors
This is a simple Nearest Neighbors algorithm built on regular python and a little bit of NumPy.

This algorithm can be used for both classification or regression problems, although, the default is set to classification.

To use this algorithm, instatiate the class with the number of neighbors you want to use to predict the output.

Such as: '''neigh = KNearestNeighbors(5)''' 

For a regression problem, set the output as regression:

'''neigh = KnearestNeighbors(n_neighbors=7, output='regression')'''
