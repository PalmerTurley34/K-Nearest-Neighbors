class KNearestNeighbors:
    import numpy as np
    def __init__(self, n_neighbors=5, output='classification'):
        self.k = n_neighbors
        self.output = output

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        if self.output == 'classification':
            self.n_classes = len(np.unique(self.y))
            # encode labels
            labels = np.unique(self.y)
            self.i_to_label = {i: label for i, label in enumerate(labels)}
            self.label_to_i = {label: i for i, label in enumerate(labels)}
            self.y_enc = np.array([self.label_to_i[label] for label in self.y])

    def single_predict(self, pred):
        # search through the training data and find the distance from each point to the target
        distances = []
        for observation in self.x:
            distances.append(np.linalg.norm(observation-pred))
              
        # look up the target values for the k nearest neighbors
        targets = list(zip(distances, self.y_enc))
        targets.sort(key=lambda x: x[0])
        # y_pred = target class with most neighbors
        k_nearest = targets[:self.k]
        if self.output == 'classification':
            votes = [0] * self.n_classes
            for vote in k_nearest:
                votes[vote[1]]+=1
            top_vote = votes.index(max(votes))
            return self.i_to_label[top_vote]
        elif self.output == 'regression':
            return np.mean(k_nearest)
    
    def predict(self, pred):
        pred = np.array(pred)
        if pred.ndim == 1:
            return self.single_predict(pred)
        elif pred.ndim == 2:
            prediction = np.array([self.single_predict(x) for x in pred])
            return prediction
    
        