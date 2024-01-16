# -*- coding: utf-8 -*-

"""BACKEND MODULE.

This module contains methods for GPU Compatiblity.

:Author: Chaithya G R <chaithyagr@gmail.com>

"""

from importlib import util

import numpy as np

from modopt.interface.errors import warn

# Handle the compatibility with variable
LIBRARIES = {
    'cupy': None,
    'tensorflow': None,
    'numpy': np,
    'torch': None,
}

CUPY_AVAILABLE = True
TORCH_AVAILABLE = True
try:
    import cupy as cp
except ImportError: # pragma: no cover
    CUPY_AVAILABLE = False
try:
    import torch
    from torch.utils.dlpack import from_dlpack as torch_from_dlpack
    from torch.utils.dlpack import to_dlpack as torch_to_dlpack
    LIBRARIES['torch'] = torch 
except ImportError: # pragma: no cover
    TORCH_AVAILABLE = False



if util.find_spec('cupy') is not None:
    try:
        import cupy as cp
        LIBRARIES['cupy'] = cp
    except ImportError:
        pass

if util.find_spec('tensorflow') is not None:
    try:
        from tensorflow.experimental import numpy as tnp
        LIBRARIES['tensorflow'] = tnp
    except ImportError:
        pass



    

def get_backend(backend):
    """Get backend.

    Returns the backend module for input specified by string.

    Parameters
    ----------
    backend: str
        String holding the backend name. One of ``'tensorflow'``,
        ``'numpy'`` or ``'cupy'``.

    Returns
    -------
    tuple
        Returns the module for carrying out calculations and the actual backend
        that was reverted towards. If the right libraries are not installed,
        the function warns and reverts to the ``'numpy'`` backend.
    """
    if backend not in LIBRARIES.keys() or LIBRARIES[backend] is None:
        msg = (
            '{0} backend not possible, please ensure that '
            + 'the optional libraries are installed.\n'
            + 'Reverting to numpy.'
        )
        warn(msg.format(backend))
        backend = 'numpy'
    return LIBRARIES[backend], backend


def get_array_module(input_data, get_string=False):
    """Get Array Module.

    This method returns the array module, which tells if the data is residing
    on GPU or CPU.

    Parameters
    ----------
    input_data : numpy.ndarray, cupy.ndarray or tf.experimental.numpy.ndarray
        Input data array
    get_string : bool, optional
        If True, returns the string corresponding to the array module. Default
        is False.

    Returns
    -------
    module
        The numpy or cupy module

    """
    if LIBRARIES['tensorflow'] is not None:
        if isinstance(input_data, LIBRARIES['tensorflow'].ndarray):
            return LIBRARIES['tensorflow'] if not get_string else 'tensorflow'
    if LIBRARIES['cupy'] is not None:
        if isinstance(input_data, LIBRARIES['cupy'].ndarray):
            return LIBRARIES['cupy'] if not get_string else 'cupy'
    if LIBRARIES['torch'] is not None:
        if LIBRARIES['torch'].is_tensor(input_data):
            return LIBRARIES['torch'] if not get_string else 'torch'
    return np if not get_string else 'numpy'


def change_backend(input_data, backend='cupy'):
    """Move data to device.

    This method changes the backend of an array. This can be used to copy data
    to GPU or to CPU.

    Parameters
    ----------
    input_data : numpy.ndarray, cupy.ndarray or tf.experimental.numpy.ndarray
        Input data array to be moved
    backend: str, optional
        The backend to use, one among ``'tensorflow'``, ``'cupy'`` and
        ``'numpy'``. Default is ``'cupy'``.

    Returns
    -------
    backend.ndarray
        An ndarray of specified backend

    """
    cur_backend = get_array_module(input_data, get_string=True)
    xp = get_array_module(input_data)
    txp, target_backend = get_backend(backend)
    if cur_backend == 'cupy' and target_backend == 'torch':
        return convert_to_tensor(input_data)
    elif cur_backend == 'torch' and target_backend == 'cupy':
        return convert_to_cupy_array(input_data)
    if xp == txp:
        return input_data
    return txp.array(input_data)


def move_to_cpu(input_data):
    """Move data to CPU.

    This method moves data from GPU to CPU. It returns the same data if it is
    already on CPU.

    Parameters
    ----------
    input_data : cupy.ndarray or tf.experimental.numpy.ndarray
        Input data array to be moved

    Returns
    -------
    numpy.ndarray
        The NumPy array residing on CPU

    Raises
    ------
    ValueError
        if the input does not correspond to any array
    """
    xp = get_array_module(input_data)

    if xp == LIBRARIES['numpy']:
        return input_data
    elif xp == LIBRARIES['cupy']:
        return input_data.get()
    elif xp == LIBRARIES['tensorflow']:
        return input_data.data.numpy()
    raise ValueError('Cannot identify the array type.')


def convert_to_tensor(input_data):
    """Convert data to a tensor.

    This method converts input data to a torch tensor. Particularly, this
    method is helpful to convert CuPy array to Tensor.

    Parameters
    ----------
    input_data : cupy.ndarray
        Input data array to be converted

    Returns
    -------
    torch.Tensor
        The tensor data

    Raises
    ------
    ImportError
        If Torch package not found

    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            'Required version of Torch package not found'
            + 'see documentation for details: https://cea-cosmic.'
            + 'github.io/ModOpt/#optional-packages',
        )

    xp = get_array_module(input_data)

    if xp == np:
        return torch.Tensor(input_data)

    return torch_from_dlpack(input_data.toDlpack()).float()


def convert_to_cupy_array(input_data):
    """Convert Tensor data to a CuPy Array.

    This method converts input tensor data to a cupy array.

    Parameters
    ----------
    input_data : torch.Tensor
        Input Tensor to be converted

    Returns
    -------
    cupy.ndarray
        The tensor data as a CuPy array

    Raises
    ------
    ImportError
        If Torch package not found

    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            'Required version of Torch package not found'
            + 'see documentation for details: https://cea-cosmic.'
            + 'github.io/ModOpt/#optional-packages',
        )

    if input_data.is_cuda:
        return cp.fromDlpack(torch_to_dlpack(input_data))

    return input_data.detach().numpy()
