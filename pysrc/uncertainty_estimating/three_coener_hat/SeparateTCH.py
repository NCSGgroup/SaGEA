import matplotlib.pyplot as plt
import numpy as np
import scipy

from TCH import TCH


class STCH():
    def __init__(self):
        self.datasets = None
        self.xn = None
        self.xm = None

        self.__r_mat = None

        pass

    def set_datasets(self, *x):
        """
        Number of datasets should be greater than or equal to 3, and the last one will be set as the preference.
        """
        self.xn = len(x)  # ...
        self.xm = len(x[0])  # ...
        assert self.xn >= 3  # ensure that ...

        self.datasets = np.array(x)

        return self

    def get_square_epsilon(self):
        var_sub = np.zeros_like(self.datasets)

        length_of_data = self.xm

        tch = TCH()
        '''get TCH var_sigma of whole ts'''
        tch.set_datasets(*self.datasets)
        var_whole = tch.get_var_epsilon()

        '''get TCH var_sigma of sub_ts'''
        full_index = np.arange(length_of_data)
        for i in range(length_of_data):
            index = np.delete(full_index, i)
            this_dataset = self.datasets[:, index]

            tch.set_datasets(*this_dataset)
            var_sub[:, i] = tch.get_var_epsilon()

        square_epsilon = np.abs(length_of_data * (var_whole[:, None]) - (length_of_data - 1) * (var_sub))

        return square_epsilon


def demo():
    """validate"""
    a, b, c, d = 1024, 2.6, 10.3, 2.2
    t = np.arange(2005, 2015, 1 / 12)
    x0 = a + b * t + c * np.sin(2 * np.pi * t) + d * np.cos(2 * np.pi * t)
    xs = []
    sigmas = np.array([2.5, 3.1, 2.1, 2.7, 0.5]) * 0.3

    epsilons = []

    for i in range(len(sigmas)):
        this_epsilon = np.random.normal(loc=0, scale=sigmas[i], size=np.shape(x0))
        this_epsilon -= np.mean(this_epsilon)

        this_x = x0 + this_epsilon

        epsilons.append(this_epsilon)
        xs.append(this_x)

    epsilons = np.array(epsilons)
    xs = np.array(xs)

    tch = TCH()
    tch.set_datasets(*xs)
    print(np.sum(epsilons ** 2, axis=1) / len(epsilons[0]))
    print(tch.get_var_epsilon())

    stch = STCH()
    stch.set_datasets(*xs)
    epsilons_tch = stch.get_square_epsilon()

    index = 0
    y_to_plot = [(epsilons[index]) ** 2, epsilons_tch[index]]

    labels = ["simulated $\epsilon^2$", "$\epsilon^2$ by Separated TCH"]

    for i in range(len(y_to_plot)):
        plt.plot(t, y_to_plot[i], label=labels[i])

    r, p = scipy.stats.pearsonr(y_to_plot[0], y_to_plot[1])
    plt.xlabel(f"year, r={'%.2f' % r}, p={'%.2f' % p}", fontsize=16)
    plt.ylabel(f"squared $\epsilon$", fontsize=16)

    plt.ylim()
    plt.legend(loc=9, fontsize=16)
    plt.show()


if __name__ == '__main__':
    demo()
