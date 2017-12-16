# -*- coding: utf-8 -*-

"""LINEAR OPERATORS

This module contains linear operator classes.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

:Version: 1.3

:Date: 19/07/2017

"""

from builtins import range, zip
import numpy as np
from modopt.base.wrappers import add_agrs_kwargs
from modopt.math.matrix import rotate
from modopt.signal.wavelet import *


class LinearParent(object):
    r"""Linear Operator Parent Class

    This class sets the structure for defining linear operator instances.

    Parameters
    ----------
    op : func
        Callable function that implements the linear operation
    adj_op : func
        Callable function that implements the linear adjoint operation

    Examples
    --------
    >>> from modopt.opt.linear import LinearParent
    >>> a = LinearParent(lambda x: x * 2, lambda x: x ** 3)
    >>> a.op(2)
    4
    >>> a.adj_op(2)
    8

    """

    def __init__(self, op, adj_op):

        self.op = op
        self.adj_op = adj_op

    def _test_operator(self, operator):
        """ Test Input Operator

        This method checks if the input operator is a callable funciton and
        adds support for `*args` and `**kwargs` if not already provided

        Parameters
        ----------
        operator : func
            Callable function

        Returns
        -------
        func wrapped by `add_agrs_kwargs`

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not callable(operator):
            raise TypeError('The input operator must be a callable function.')

        return add_agrs_kwargs(operator)

    @property
    def op(self):
        """Linear Operator

        This method defines the linear operator

        """

        return self._op

    @op.setter
    def op(self, operator):

        self._op = self._test_operator(operator)

    @property
    def adj_op(self):
        """Linear Adjoint Operator

        This method defines the linear operator

        """

        return self._adj_op

    @adj_op.setter
    def adj_op(self, operator):

        self._adj_op = self._test_operator(operator)


class Identity(LinearParent):
    """Identity operator class

    This is a dummy class that can be used in the optimisation classes

    """

    def __init__(self):

        self.l1norm = 1.0
        self.op = lambda x: x
        self.adj_op = self.op


class Wavelet(LinearParent):
    """Wavelet class

    This class defines the wavelet transform operators

    Parameters
    ----------
    data : np.ndarray
        Input data array, normally an array of 2D images
    wavelet_opt: str, optional
        Additional options for `mr_transform`

    """

    def __init__(self, data, wavelet_opt=''):

        self.y = data
        self.data_shape = data.shape[-2:]
        n = data.shape[0]

        self.filters = get_mr_filters(self.data_shape, opt=wavelet_opt)
        self.l1norm = n * np.sqrt(sum((np.sum(np.abs(filter)) ** 2 for
                                       filter in self.filters)))
        self.op = lambda x: filter_convolve_stack(x, self.filters)
        self.adj_op = lambda x: filter_convolve_stack(x, self.filters,
                                                      filter_rot=True)


class LinearCombo(LinearParent):
    r"""Linear combination class

    This class defines a combination of linear transform operators

    Parameters
    ----------
    operators : list, tuple or np.ndarray
        List of linear operator class instances
    weights : list, tuple or np.ndarray
        List of weights for combining the linear adjoint operator results

    Examples
    --------
    >>> from modopt.opt.linear import LinearCombo
    >>> a = LinearParent(lambda x: x * 2, lambda x: x ** 3)
    >>> b = LinearParent(lambda x: x * 4, lambda x: x ** 5)
    >>> c = LinearCombo([a, b])
    >>> a.op(2)
    4
    >>> b.op(2)
    8
    >>> c.op(2)
    array([4, 8], dtype=object)
    >>> a.adj_op(2)
    8
    >>> b.adj_op(2)
    32
    >>> c.adj_op([2, 2])
    20.0

    """

    def __init__(self, operators, weights=None):

        operators, weights = self._check_inputs(operators, weights)
        self.operators = operators
        self.weights = weights
        self.op = self._op_method
        self.adj_op = self._adj_op_method

    def _check_type(self, input_val):
        """ Check Input Type

        This method checks if the input is a list, tuple or a numpy array and
        converts the input to a numpy array

        Parameters
        ----------
        input_val : list, tuple or np.ndarray

        Returns
        -------
        np.ndarray of input

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not isinstance(input_val, (list, tuple, np.ndarray)):
            raise TypeError('Invalid input type, input must be a list, tuple '
                            'or numpy array.')

        return np.array(input_val)

    def _check_inputs(self, operators, weights):
        """ Check Inputs

        This method cheks that the input operators and weights are correctly
        formatted

        Parameters
        ----------
        operators : list, tuple or np.ndarray
            List of linear operator class instances
        weights : list, tuple or np.ndarray
            List of weights for combining the linear adjoint operator results

        Returns
        -------
        tuple operators and weights

        Raises
        ------
        ValueError
            If the number of weights does not match the number of operators
        TypeError
            If the individual weight values are not floats

        """

        operators = self._check_type(operators)

        if not isinstance(weights, type(None)):

            weights = self._check_type(weights)

            if weights.size != operators.size:
                raise ValueError('The number of weights must match the '
                                 'number of operators.')

            if not np.issubdtype(weights.dtype, float):
                raise TypeError('The weights must be a list of float values.')

        return operators, weights

    def _op_method(self, data):
        """Operator

        This method returns the input data operated on by all of the operators

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray linear operation results

        """

        res = np.empty(len(self.operators), dtype=np.ndarray)

        for i in range(len(self.operators)):
            res[i] = self.operators[i].op(data)

        return res

    def _adj_op_method(self, data):
        """Adjoint operator

        This method returns the combination of the result of all of the
        adjoint operators. If weights are provided the comibination is the sum
        of the weighted results, otherwise the combination is the mean.

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Returns
        -------
        np.ndarray adjoint operation results

        """

        if isinstance(self.weights, type(None)):

            return np.mean([operator.adj_op(x) for x, operator in
                           zip(data, self.operators)], axis=0)

        else:

            return np.sum([weight * operator.adj_op(x) for x, operator,
                          weight in zip(data, self.operators, self.weights)],
                          axis=0)
