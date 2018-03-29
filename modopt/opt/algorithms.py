# -*- coding: utf-8 -*-

r"""OPTIMISATION ALGOTITHMS

This module contains class implementations of various optimisation algoritms.

:Author: Samuel Farrens <samuel.farrens@cea.fr>

NOTES
-----

Input classes must have the following properties:

    * **Gradient Operators**

    Must have the following methods:

        * ``get_grad()`` - calculate the gradient

    Must have the following variables:

        * ``grad`` - the gradient

    * **Linear Operators**

    Must have the following methods:

        * ``op()`` - operator
        * ``adj_op()`` - adjoint operator

    * **Proximity Operators**

    Must have the following methods:

        * ``op()`` - operator

The following notation is used to implement the algorithms:

    * x_old is used in place of :math:`x_{n}`.
    * x_new is used in place of :math:`x_{n+1}`.
    * x_prox is used in place of :math:`\tilde{x}_{n+1}`.
    * x_temp is used for intermediate operations.

"""

from __future__ import division, print_function
from builtins import range, zip
import numpy as np
from inspect import getmro
from modopt.interface.errors import warn
from modopt.opt.cost import costObj
from modopt.opt.linear import Identity

# Package import
from ..base.observable import Observable, MetricObserver


class SetUp(Observable):
    """Algorithm Set-Up

    This class contains methods for checking the set-up of an optimisation
    algotithm and produces warnings if they do not comply

    """

    def __init__(self, metric_call_period=5, metrics={}, verbose=False):
        print('Alt version of Modopt')

        self.converge = False
        self.verbose = verbose

        self._op_parents = ('GradParent', 'ProximityParent', 'LinearParent',
                            'costObj')
        self.metric_call_period = metric_call_period

        Observable.__init__(self, ["cv_metrics"])
        for name, dic in metrics.items():
            observer = MetricObserver(name, dic['metric'],
                                      dic['mapping'],
                                      dic['cst_kwargs'],
                                      dic['early_stopping'])
            self.add_observer("cv_metrics", observer)


    def _check_input_data(self, data):
        """ Check Input Data Type

        This method checks if the input data is a numpy array

        Parameters
        ----------
        data : np.ndarray
            Input data array

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not isinstance(data, np.ndarray):
            raise TypeError('Input data must be a numpy array.')

    def _check_param(self, param):
        """ Check Algorithm Parameters

        This method checks if the specified algorithm parameters are floats

        Parameters
        ----------
        param : float
            Parameter value

        Raises
        ------
        TypeError
            For invalid input type

        """

        if not isinstance(param, float):
            raise TypeError('Algorithm parameter must be a float value.')

    def _check_param_update(self, param_update):
        """ Check Algorithm Parameters

        This method checks if the specified algorithm parameters are floats

        Parameters
        ----------
        param_update : function
            Callable function

        Raises
        ------
        TypeError
            For invalid input type

        """

        if (not isinstance(param_update, type(None)) and
                not callable(param_update)):
            raise TypeError('Algorithm parameter update must be a callabale '
                            'function.')

    def _check_operator(self, operator):
        """ Check Set-Up

        This method checks algorithm operator against the expected parent
        classes

        Parameters
        ----------
        operator : str
            Algorithm operator to check

        """

        if not isinstance(operator, type(None)):
            tree = [obj.__name__ for obj in getmro(operator.__class__)]

            if not any([parent in tree for parent in self._op_parents]):
                warn('{0} does not inherit an operator '
                     'parent.'.format(str(operator.__class__)))


class FISTA(object):
    r"""FISTA

    This class is inhereited by optimisation classes to speed up convergence

    """

    def __init__(self):

        self._t_now = 1.0
        self._t_prev = 1.0

    def update_lambda(self, *args, **kwargs):
        r"""Update lambda

        This method updates the value of lambda

        Returns
        -------
        float current lambda value

        Notes
        -----
        Implements steps 3 and 4 from algoritm 10.7 in [B2011]_

        """

        # Steps 3 and 4 from alg.10.7.
        self._t_prev = self._t_now
        self._t_now = (1 + np.sqrt(4 * self._t_prev ** 2 + 1)) * 0.5

        return 1 + (self._t_prev - 1) / self._t_now


class ForwardBackward(SetUp):
    r"""Forward-Backward optimisation

    This class implements standard forward-backward optimisation with an the
    option to use the FISTA speed-up

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    grad : class
        Gradient operator class
    prox : class
        Proximity operator class
    cost : class or str, optional
        Cost function class (default is 'auto'); Use 'auto' to automatically
        generate a costObj instance
    beta_param : float, optional
        Initial value of the beta parameter (default is 1.0)
    lambda_param : float, optional
        Initial value of the lambda parameter (default is 1.0)
    beta_update : function, optional
        Beta parameter update method (default is None)
    lambda_update : function or string, optional
        Lambda parameter update method (default is 'fista')
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, grad, prox, cost='auto', beta_param=1.0,
                 lambda_param=1.0, beta_update=None, lambda_update='fista',
                 auto_iterate=True):

        # Set default algorithm properties
        super(ForwardBackward, self).__init__()

        # Set the initial variable values
        self._check_input_data(x)
        self._x_old = np.copy(x)
        self._z_old = np.copy(x)

        # Set the algorithm operators
        (self._check_operator(operator) for operator in (grad, prox, cost))
        self._grad = grad
        self._prox = prox
        if cost == 'auto':
            self._cost_func = costObj([self._grad, self._prox])
        else:
            self._cost_func = cost

        # Set the algorithm parameters
        (self._check_param(param) for param in (beta_param, lambda_param))
        self._beta = beta_param
        self._lambda = lambda_param

        # Set the algorithm parameter update methods
        if isinstance(lambda_update, str) and lambda_update == 'fista':
            self._lambda_update = FISTA().update_lambda
        else:
            self._check_param_update(lambda_update)
            self._lambda_update = lambda_update
        self._check_param_update(beta_update)
        self._beta_update = beta_update

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _update_param(self):
        r"""Update parameters

        This method updates the values of the algorthm parameters with the
        methods provided

        """

        # Update the gamma parameter.
        if not isinstance(self._beta_update, type(None)):
            self._beta = self._beta_update(self._beta)

        # Update lambda parameter.
        if not isinstance(self._lambda_update, type(None)):
            self._lambda = self._lambda_update(self._lambda)

    def _update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 10.7 (or 10.5) from [B2011]_

        """

        # Step 1 from alg.10.7.
        self._grad.get_grad(self._z_old)
        y_old = self._z_old - self._beta * self._grad.grad

        # Step 2 from alg.10.7.
        self._x_new = self._prox.op(y_old, extra_factor=self._beta)

        # Step 5 from alg.10.7.
        self._z_new = self._x_old + self._lambda * (self._x_new - self._x_old)

        # Update old values for next iteration.
        np.copyto(self._x_old, self._x_new)
        np.copyto(self._z_old, self._z_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for i in range(max_iter):
            self._update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self._z_new


class GenForwardBackward(SetUp):
    r"""Generalized Forward-Backward Algorithm

    This class implements algorithm 1 from [R2012]_

    Parameters
    ----------
    x : list, tuple or np.ndarray
        Initial guess for the primal variable
    grad : class instance
        Gradient operator class
    prox_list : list
        List of proximity operator class instances
    cost : class or str, optional
        Cost function class (default is 'auto'); Use 'auto' to automatically
        generate a costObj instance
    gamma_param : float, optional
        Initial value of the gamma parameter (default is 1.0)
    lambda_param : float, optional
        Initial value of the lambda parameter (default is 1.0)
    gamma_update : function, optional
        Gamma parameter update method (default is None)
    lambda_update : function, optional
        Lambda parameter parameter update method (default is None)
    weights : list, tuple or np.ndarray, optional
        Proximity operator weights (default is None)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, grad, prox_list, cost='auto', gamma_param=1.0,
                 lambda_param=1.0, gamma_update=None, lambda_update=None,
                 weights=None, auto_iterate=True):

        # Set default algorithm properties
        super(GenForwardBackward, self).__init__()

        # Set the initial variable values
        self._check_input_data(x)
        self._x_old = np.copy(x)

        # Set the algorithm operators
        (self._check_operator(operator) for operator in [grad, cost]
         + prox_list)
        self._grad = grad
        self._prox_list = np.array(prox_list)
        if cost == 'auto':
            self._cost_func = costObj([self._grad] + prox_list)
        else:
            self._cost_func = cost

        # Set the algorithm parameters
        (self._check_param(param) for param in (gamma_param, lambda_param))
        self._gamma = gamma_param
        self._lambda_param = lambda_param

        # Set the algorithm parameter update methods
        (self._check_param_update(param_update) for param_update in
         (gamma_update, lambda_update))
        self._gamma_update = gamma_update
        self._lambda_update = lambda_update

        # Set the proximity weights
        self._set_weights(weights)

        # Set initial z
        self._z = np.array([self._x_old for i in range(self._prox_list.size)])

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _set_weights(self, weights):
        """ Set Weights

        This method sets weights on each of the proximty operators provided

        Parameters
        ----------
        weights : list, tuple or np.ndarray
            List of weights

        Raises
        ------
        TypeError
            For invalid input type
        ValueError
            If weights do not sum to one

        """

        if isinstance(weights, type(None)):
            weights = np.repeat(1.0 / self._prox_list.size,
                                self._prox_list.size)
        elif not isinstance(weights, (list, tuple, np.ndarray)):
            raise TypeError('Weights must be provided as a list.')

        weights = np.array(weights)

        if not np.issubdtype(weights.dtype, np.floating):
            raise ValueError('Weights must be list of float values.')

        if weights.size != self._prox_list.size:
            raise ValueError('The number of weights must match the number of '
                             'proximity operators.')

        if np.sum(weights) != 1.0:
            raise ValueError('Proximity operator weights must sum to 1.0.'
                             'Current sum of weights = ' +
                             str(np.sum(weights)))

        self._weights = weights

    def _update_param(self):
        r"""Update parameters

        This method updates the values of the algorthm parameters with the
        methods provided

        """

        # Update the gamma parameter.
        if not isinstance(self._gamma_update, type(None)):
            self._gamma = self._gamma_update(self._gamma)

        # Update lambda parameter.
        if not isinstance(self._lambda_update, type(None)):
            self._lambda_param = self._lambda_update(self._lambda_param)

    def _update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements algorithm 1 from [R2012]_

        """

        # Calculate gradient for current iteration.
        self._grad.get_grad(self._x_old)

        # Update z values.
        for i in range(self._prox_list.size):
            z_temp = (2 * self._x_old - self._z[i] - self._gamma *
                      self._grad.grad)
            z_prox = self._prox_list[i].op(z_temp, extra_factor=self._gamma /
                                           self._weights[i])
            self._z[i] += self._lambda_param * (z_prox - self._x_old)

        # Update current reconstruction.
        self._x_new = np.sum((z_i * w_i for z_i, w_i in
                              zip(self._z, self._weights)), axis=0)

        # Update old values for next iteration.
        np.copyto(self._x_old, self._x_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        for i in range(max_iter):
            self._update()

            if self.converge:
                print(' - Converged!')
                break

        self.x_final = self._x_new


class Condat(SetUp):
    r"""Condat optimisation

    This class implements algorithm 10.7 from [Con2013]_

    Parameters
    ----------
    x : np.ndarray
        Initial guess for the primal variable
    y : np.ndarray
        Initial guess for the dual variable
    grad : class instance
        Gradient operator class
    prox : class instance
        Proximity primal operator class
    prox_dual : class instance
        Proximity dual operator class
    linear : class instance, optional
        Linear operator class (default is None)
    cost : class or str, optional
        Cost function class (default is 'auto'); Use 'auto' to automatically
        generate a costObj instance
    rho : float, optional
        Relaxation parameter (default is 0.5)
    sigma : float, optional
        Proximal dual parameter (default is 1.0)
    tau : float, optional
        Proximal primal paramater (default is 1.0)
    rho_update : function, optional
        Relaxation parameter update method (default is None)
    sigma_update : function, optional
        Proximal dual parameter update method (default is None)
    tau_update : function, optional
        Proximal primal parameter update method (default is None)
    auto_iterate : bool, optional
        Option to automatically begin iterations upon initialisation (default
        is 'True')

    """

    def __init__(self, x, y, grad, prox, prox_dual, linear=None, cost='auto',
                 rho=0.5,  sigma=1.0, tau=1.0, rho_update=None,
                 sigma_update=None, tau_update=None, auto_iterate=True,
                 metric_call_period=5, metrics={}):

        # Set default algorithm properties
        super(Condat, self).__init__()

        # Set the initial variable values
        (self._check_input_data(data) for data in (x, y))
        self._x_old = np.copy(x)
        self._y_old = np.copy(y)

        # Set the algorithm operators
        (self._check_operator(operator) for operator in (grad, prox, prox_dual,
         linear, cost))
        self._grad = grad
        self._prox = prox
        self._prox_dual = prox_dual
        if isinstance(linear, type(None)):
            self._linear = Identity()
        else:
            self._linear = linear
        if cost == 'auto':
            self._cost_func = costObj([self._grad, self._prox,
                                       self._prox_dual])
        else:
            self._cost_func = cost

        # Set the algorithm parameters
        (self._check_param(param) for param in (rho, sigma, tau))
        self._rho = rho
        self._sigma = sigma
        self._tau = tau

        # Set the algorithm parameter update methods
        (self._check_param_update(param_update) for param_update in
         (rho_update, sigma_update, tau_update))
        self._rho_update = rho_update
        self._sigma_update = sigma_update
        self._tau_update = tau_update

        # Automatically run the algorithm
        if auto_iterate:
            self.iterate()

    def _update_param(self):
        r"""Update parameters

        This method updates the values of the algorthm parameters with the
        methods provided

        """

        # Update relaxation parameter.
        if not isinstance(self._rho_update, type(None)):
            self._rho = self._rho_update(self._rho)

        # Update proximal dual parameter.
        if not isinstance(self._sigma_update, type(None)):
            self._sigma = self._sigma_update(self._sigma)

        # Update proximal primal parameter.
        if not isinstance(self._tau_update, type(None)):
            self._tau = self._tau_update(self._tau)

    def _update(self):
        r"""Update

        This method updates the current reconstruction

        Notes
        -----
        Implements equation 9 (algorithm 3.1) from [Con2013]_

        - primal proximity operator set up for positivity constraint

        """
        # Step 1 from eq.9.
        self._grad.get_grad(self._x_old)

        x_prox = self._prox.op(self._x_old - self._tau * self._grad.grad -
                               self._tau * self._linear.adj_op(self._y_old))

        # Step 2 from eq.9.
        y_temp = (self._y_old + self._sigma *
                  self._linear.op(2 * x_prox - self._x_old))

        y_prox = (y_temp - self._sigma * self._prox_dual.op(y_temp /
                  self._sigma, extra_factor=(1.0 / self._sigma)))

        # Step 3 from eq.9.
        self._x_new = self._rho * x_prox + (1 - self._rho) * self._x_old
        self._y_new = self._rho * y_prox + (1 - self._rho) * self._y_old

        del x_prox, y_prox, y_temp

        # Update old values for next iteration.
        np.copyto(self._x_old, self._x_new)
        np.copyto(self._y_old, self._y_new)

        # Update parameter values for next iteration.
        self._update_param()

        # Test cost function for convergence.
        if self._cost_func:
            self.converge = self._cost_func.get_cost(self._x_new, self._y_new)

    def iterate(self, max_iter=150):
        r"""Iterate

        This method calls update until either convergence criteria is met or
        the maximum number of iterations is reached

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of iterations (default is ``150``)

        """

        print('toto')
        # exit(0)
        for i in range(max_iter):
            self._update()

            if self.converge:
                print(' - Converged!')
                break
            # metric computation and early-stopping check
            if self.idx % self.metric_call_period == 0:
                kwargs = self.get_notify_observers_kwargs()
                self.notify_observers('cv_metrics', **kwargs)
                if self.any_convergence_flag():
                    if self.verbose:
                        print("\n-----> early-stopping done")
                    break

        # retrieve metrics results
        self.retrieve_outputs()
        # rename outputs as attributes
        self.x_final = self._x_new
        self.y_final = self._y_new

    def get_notify_observers_kwargs(self):
        """ Return the mapping between the metrics call and the iterated
        variables.

        Return
        ----------
        notify_observers_kwargs: dict,
           the mapping between the iterated variables.
        """
        return {'x_new': self.x_new, 'y_new':self.y_new, 'idx':self.idx}

    def retrieve_outputs(self):
        """ Declare the outputs of the algorithms as attributes: x_final,
        y_final, metrics.
        """
        print('Im here')
        metrics = {}
        for obs in self._observers['cv_metrics']:
            metrics[obs.name] = obs.retrieve_metrics()
        self.metrics = metrics
