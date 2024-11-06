import warnings

import numpy as np


class MonteCarloConfig:
    def __init__(self):
        self.__sample_num = 1000

    @property
    def sample_num(self):
        return self.__sample_num

    @sample_num.setter
    def sample_num(self, sample_num):
        self.__sample_num = sample_num

    def set_sample_num(self, sample_num):
        self.__sample_num = sample_num
        return self


class MonteCarlo:
    def __init__(self):
        self.configuration = MonteCarloConfig()

        self.__cov_mat = None
        self.__data_length = None
        pass

    def set_input(self, cov_mat):
        input_shape = np.shape(cov_mat)
        assert len(input_shape) == 2
        assert input_shape[0] == input_shape[1]

        self.__cov_mat = cov_mat
        self.__data_length = input_shape[0]

        return self

    def get_noise(self):
        """
        return: SHC
        """
        cov_mat = self.__cov_mat
        noise_length = self.__data_length

        assert cov_mat is not None
        assert noise_length is not None

        noise = np.random.multivariate_normal(
            mean=np.array([0] * noise_length),
            cov=cov_mat,
            size=self.configuration.sample_num
        )

        return noise

    @staticmethod
    def get_covariance(data: np.ndarray, at_index: str or tuple = None, block_size=None, compress=np.float16):
        """
        :param data: 2(+)-dimensional numpy array in shape of (n: number of samples, *m: value)
        :param at_index: "full", "diag" or a tuple denoting index of data
        :param block_size: int or None,
        :param compress: use lower precision data types during calculations of full var-cov matrix
        :return:
            if length of data == 1, variance of data;
            else:
                if at_index == "full", full var-cov matrix in shape of (*m, *m);
                if at_index == "diag", variance in shape of (*m,);
                if at_index is tuple, covariance at given index with all other data, in shape of (*m)
        """
        assert len(data.shape) >= 1
        if len(data.shape) == 1:
            return np.var(data)

        if at_index is None:
            at_index = "diag"
        assert at_index in ("full", "diag") or isinstance(at_index, tuple)

        value_length = data.shape[0]
        value_shape = data.shape[1:]

        if at_index == "full":
            warnings.warn("experimental for now, may cause leakage and overflow")

            data = data.astype(compress)

            if block_size is None:
                reshaped_data = data.reshape(value_length, -1).astype(compress)
                covariance_matrix = np.cov(reshaped_data, rowvar=False)

                output_shape = (*value_shape, *value_shape)
                covariance_matrix = covariance_matrix.reshape(output_shape)

            else:
                count = 0
                dim = np.prod(data.shape[1:])

                mean = np.zeros(dim, dtype=compress)
                M2 = np.zeros((dim, dim), dtype=compress)

                for start in range(0, value_length, block_size):
                    print(start)

                    end = min(start + block_size, value_length)
                    block = data[start:end].reshape(end - start, -1)

                    block_mean = np.mean(block, axis=0)
                    count += block.shape[0]

                    delta = block_mean - mean
                    mean += delta * (block.shape[0] / count)

                    M2 += np.einsum('ij,ik->jk', block - block_mean, block - mean)

                covariance_matrix = M2 / (count - 1)

                covariance_matrix = covariance_matrix.reshape(*data.shape[1:], *data.shape[1:])

            return covariance_matrix

        elif at_index == "diag":
            return np.var(data, axis=0)

        elif isinstance(at_index, tuple):
            assert len(at_index) == len(value_shape)

            index_mapping = {}

            for index in np.ndindex(value_shape):
                flat_index = np.ravel_multi_index(index, value_shape)
                index_mapping[flat_index] = index

            reshaped_data = data.reshape(value_length, -1)

            get_key = lambda d, v: [k for k, v in d.items() if v == at_index][0]
            index1d = get_key(index_mapping, at_index)

            return (reshaped_data[None, :, index1d] @ reshaped_data).reshape(value_shape)

        else:
            assert False
