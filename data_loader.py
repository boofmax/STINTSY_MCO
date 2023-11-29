import numpy as np


class DataLoader(object):

    def __init__(self, X, y, batch_size):
        """Class constructor for DataLoader

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) containing the
            data; there are N samples each of dimension D.
            y {np.ndarray} -- A numpy array of shape (N, 1) containing the
            ground truth values.
            batch_size {int} -- An integer representing the number of instances
            per batch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size

        self.indices = np.array([i for i in range(self.X.shape[0])])
        np.random.seed(1)

    def shuffle(self):
        """Shuffles the indices in self.indices.
        """

        # TODO: Use np.random.shuffle() to shuffles the indices in self.indices
        np.random.shuffle(self.indices)

    def get_batch(self, mode='train'):
        """Returns self.X and self.y divided into different batches of size
        self.batch_size according to the shuffled self.indices.

        Arguments:
            mode {str} -- A string which determines the mode of the model. This
            can either be `train` or `test`.

        Returns:
            list, list -- List of np.ndarray containing the data divided into
            different batches of size self.batch_size; List of np.ndarray
            containing the ground truth labels divided into different batches
            of size self.batch_size
        """

        X_batch = []
        y_batch = []

        # TODO: If mode is set to `train`, shuffle the indices first using
        # self.shuffle().
        if mode == 'train':
            self.shuffle()
        elif mode == 'test':
            self.indices = np.array([i for i in range(self.X.shape[0])])

        # The loop that will iterate from 0 to the number of instances with
        # step equal to self.batch_size
        for i in range(0, len(self.indices), self.batch_size):

            # TODO: Check if we can still get self.batch_size from the
            # remaining indices starting from index i. Edit the condition
            # below.
            if (len(self.indices) - i) >= self.batch_size:
                indices = self.indices[i:i + self.batch_size]

            # TODO: Else, just get the remaining indices from index i until the
            # last element in the list. Edit the statement inside the else
            # block.
            else:
                indices = self.indices[i:len(self.indices)]

            X_batch.append(self.X[indices])
            y_batch.append(self.y[indices])

        return X_batch, y_batch
