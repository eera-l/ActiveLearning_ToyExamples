import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class Sampler(object):
    def __init__(self, classifier='SVM', cluster_type='moons', informativeness_measure='least_confidence',
                 sample_size=1000, train_size=30, first_samples_size=15, centers=None,
                 clusters_std=None, noise=0.0, plot=False, random_state=12):
        assert first_samples_size <= train_size
        self.classifier = classifier
        self.cluster_type = cluster_type
        self.informativeness_measure = informativeness_measure
        self.sample_size = sample_size
        self.train_size = train_size
        self.centers = centers
        self.clusters_std = clusters_std
        self.noise = noise
        self.first_samples_size = first_samples_size
        self.plot = plot
        self.random_state = random_state

    def __generate_data__(self):
        """
        Generates data in clusters of optional shape and distribution
        :return: Array of data points and corresponding cluster label
        """
        if self.cluster_type == 'moons' or self.cluster_type == 'circles':
            if self.informativeness_measure != 'least_confidence':
                print('Circles and moons can only be implemented with binary classification.' +
                      '\nInformativeness measure has been changed to least confidence.')
                self.informativeness_measure = 'least_confidence'
        if self.cluster_type == 'blobs':
            X, y = make_blobs(n_samples=self.sample_size,
                              centers=self.centers, cluster_std=self.clusters_std,
                              n_features=2, random_state=self.random_state)
        elif self.cluster_type == 'moons':
            X, y = make_moons(n_samples=self.sample_size, noise=self.noise,
                              random_state=self.random_state)
        elif self.cluster_type == 'circles':
            X, y = make_circles(n_samples=self.sample_size, noise=self.noise,
                                random_state=self.random_state)
        return X, y

    def __split_data__(self, X, y):
        """
        Splits data into training and test set
        :return: samples and labels for training set,
        samples and labels for test set
        """
        test_size = X.shape[0] - self.train_size
        try:
            x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state,
                                                                test_size=test_size)
        except ValueError:
            print(test_size)
            print(X.shape[0])
        return x_train, x_test, y_train, y_test

    def __fit_classifier__(self, x_train, y_train):
        """
        Fit an SVM or logistic regression model
        :return: trained classifier
        """
        if self.classifier == 'SVM':
            clf = SVC(random_state=self.random_state, probability=True)
        elif self.classifier == 'LR':
            clf = LogisticRegression(random_state=self.random_state)
        clf.fit(x_train, y_train)
        return clf

    def __random_sampling__(self, X, y):
        """
        Trains and evaluates classifier according to random sampling
        :return: trained classifier, accuracy score
        """
        x_train, x_test, y_train, y_test = self.__split_data__(X, y)
        clf = self.__fit_classifier__(x_train, y_train)
        score = clf.score(x_test, y_test)
        return clf, score

    def __uncertainty_sampling__(self, X, y):
        """
        First trains the classifier on the specified set of starting samples,
        then predicts on the test set. Then picks the specified number
        of most informative samples (i.e., where the probability is closest to 0.5)
        and retrains the classifier on those samples
        :return: trained classifier, accuracy score
        """
        x_train, x_test, y_train, y_test = self.__split_data__(X, y)
        # Train on the first n samples the first time
        clf = self.__fit_classifier__(x_train[:self.first_samples_size],
                                     y_train[:self.first_samples_size])
        # Predict probabilities on test data
        preds = clf.predict_proba(x_test)
        clf, score = self.__rank_samples__(preds, x_train, y_train, x_test, y_test)
        return clf, score

    def __rank_samples__(self, preds, x_train, y_train, x_test, y_test):
        num_classes = preds.shape[1]
        end_index = num_classes + 2
        # Make array of probabilities and corresponding samples and labels
        preds = np.hstack((preds, x_test, y_test.reshape(-1, 1)))
        if self.informativeness_measure == 'least_confidence':
            # x^star = argmax 1 - P_theta(y_hat | x)
            preds = np.array(sorted(preds, key=lambda x: 1 - max(x[:num_classes]), reverse=True))
        elif self.informativeness_measure == 'margin':
            # x^star = argmin P_theta(y1_hat | x) - P_theta(y2_hat | x)
            preds = np.array(sorted(preds, key=lambda x: max(x[:num_classes]) - sorted(x[:num_classes], reverse=True)[1]))
        elif self.informativeness_measure == 'entropy':
            # x^star = argmax - sum(P_theta(y_i | x) * log P_theta(y_i | x))
            preds = np.array(sorted(preds, key=lambda x: -np.sum([x[i] * np.log2(x[i]) for i in range(num_classes)]),
                                    reverse=True))
        # Pick the most informative samples together with the samples the classifier has
        # already been trained on. For cumulative training
        x_train = np.vstack((preds[:self.train_size - self.first_samples_size][:, num_classes:end_index],
                             x_train[:self.first_samples_size]))
        y_train = np.hstack((preds[:self.train_size - self.first_samples_size][:, end_index].flatten(),
                             y_train[:self.first_samples_size]))
        x_test = np.vstack((preds[self.train_size - self.first_samples_size:][:, num_classes:end_index],
                            x_train[self.first_samples_size:]))
        y_test = np.hstack((preds[self.train_size - self.first_samples_size:][:, end_index].flatten(),
                            y_train[self.first_samples_size:]))
        # Retrain the classifier on these new samples
        clf = self.__fit_classifier__(x_train, y_train)
        score = clf.score(x_test, y_test)
        return clf, score

    def __make_meshgrid__(self, x, y, h=.02):
        """
        From: https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface
        :param x: array in the shape (n, 1) for the first feature
        :param y: array in the shape (n, 1) for the second feature
        :param h: step in np.arange
        :return: meshgrid of x and y coordinates
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def __plot_contours__(self, ax, clf, xx, yy, **params):
        """
        From: https://stackoverflow.com/questions/51297423/plot-scikit-learn-sklearn-svm-decision-boundary-surface
        :param ax: axis in the plot
        :param clf: trained classifier
        :param xx: x coordinates of the meshgrid
        :param yy: y coordinates of the meshgrid
        :param params: params for plot
        """
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, **params)

    def __call__(self):
        """
        Generates the data, performs classification with random and uncertainty sampling
        :return: accuracy score of random and uncertainty sampling
        """
        X, y = self.__generate_data__()
        rs_clf, rs_score = self.__random_sampling__(X, y)
        us_clf, us_score = self.__uncertainty_sampling__(X, y)
        if self.plot:
            X0, X1 = X[:, 0], X[:, 1]
            xx, yy = self.__make_meshgrid__(X0, X1)
            fig = plt.figure(figsize=(7, 3))
            ax0 = fig.add_subplot(121)
            self.__plot_contours__(ax0, rs_clf, xx, yy, cmap=plt.cm.winter, alpha=0.5)
            ax0.scatter(X[:, 0], X[:, 1], c=y)
            ax0.set_title(f'Random Sampling\nAccuracy: {rs_score * 100:.1f}%')
            ax1 = fig.add_subplot(122)
            self.__plot_contours__(ax1, us_clf, xx, yy, cmap=plt.cm.winter, alpha=0.5)
            ax1.scatter(X[:, 0], X[:, 1], c=y)
            ax1.set_title(f'Uncertainty Sampling\nAccuracy: {us_score * 100:.1f}%')
            plt.show()
        return rs_score, us_score


def plot_learning_curve(sample_size, train_size_ratio, first_samples_ratio=2, n_attempts=100):
    """
    For the specified number of iterations, trains the classifier on n samples
    as n = (i * samplesize) / train_size_ratio. If train_size_ratio and n_attempts are set to 100,
    the classifier will be trained on 1% to 99% of the data, with an increase of 1% each time.
    If train_size_ratio is set to 1000 and n_attempts is set to 100, the classifier will be trained on
    0.1% to 9.9% of the data, and so on.
    :param sample_size: total number of samples to generate
    :param train_size_ratio: magnitude of the fraction of data to use for training. E.g. 100 = start from 1%,
    1000 = start from 0.1%
    :param first_samples_ratio: ratio of the training samples to be used for the initial training of the classifier.
    :param n_attempts: number of steps to train the classifier on. At each step the classifier is reinitialized
    and retrained in an increasing fraction of data.
    """
    al_scores = []
    r_scores = []
    for i in range(1, n_attempts):
        train_size = int(i * sample_size / train_size_ratio)
        sampler = Sampler(classifier='SVM', cluster_type='moons', informativeness_measure='least_confidence',
                          random_state=i, train_size=train_size,
                          noise=.15, first_samples_size=int(train_size / first_samples_ratio),
                          sample_size=sample_size)
        rs_score, us_score = sampler()
        al_scores.append(us_score * 100)
        r_scores.append(rs_score * 100)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    x = np.linspace(sample_size / (train_size_ratio * 100), sample_size / train_size_ratio * (n_attempts / 100),
                    num=n_attempts - 1)
    ax.plot(x, al_scores, label='Active learning')
    ax.plot(x, r_scores, label='Random sampling')
    ax.set_xlabel('Fraction of data for training')
    ax.set_ylabel('Test accuracy')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    counter = 0
    n_attempts = 100
    for i in range(n_attempts):
        sampler = Sampler(classifier='LR', cluster_type='blobs', informativeness_measure='margin', random_state=i,
                          train_size=20, clusters_std=[1.4, 1, 1.2], centers=([-2, 0], [0, 2], [2, 4]),
                          first_samples_size=10)
        rs_score, us_score = sampler()
        if us_score > rs_score:
            counter += 1
    print(f'Percentage of times uncertainty sampling performed better than random, LR: '
          f'{counter / n_attempts * 100:.2f}%')
    counter = 0
    for i in range(1, n_attempts):
        sampler = Sampler(classifier='SVM', cluster_type='moons', informativeness_measure='least_confidence',
                          random_state=i, train_size=int(i * 10000 / 1000),
                          noise=.15, first_samples_size=int((i * 10000 / 1000) / 2))
        rs_score, us_score = sampler()
        if us_score > rs_score:
            counter += 1
    print(f'Percentage of times uncertainty sampling performed better than random, SVM with moons: '
          f'{counter / n_attempts * 100:.2f}%')
    counter = 0
    for i in range(n_attempts):
        sampler = Sampler(classifier='SVM', cluster_type='circles', informativeness_measure='least_confidence',
                          random_state=i, train_size=30,
                          first_samples_size=15)
        rs_score, us_score = sampler()
        if us_score > rs_score:
            counter += 1
    print(f'Percentage of times uncertainty sampling performed better than random, SVM with circles: '
          f'{counter / n_attempts * 100:.2f}%')
    plot_learning_curve(sample_size=10000, train_size_ratio=1000, first_samples_ratio=2, n_attempts=1000)
