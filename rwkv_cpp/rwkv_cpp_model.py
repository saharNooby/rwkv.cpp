import os
import multiprocessing
from . import rwkv_cpp_shared_library
from typing import Tuple, Optional
import numpy as np

class RWKVModel:
    """
    PyTorch wrapper around rwkv.cpp model.
    """

    def __init__(
            self,
            shared_library: rwkv_cpp_shared_library.RWKVSharedLibrary,
            model_path: str,
            thread_count: int = max(1, multiprocessing.cpu_count() // 2)
    ):
        """
        Loads the model and prepares it for inference.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        shared_library : RWKVSharedLibrary
            rwkv.cpp shared library.
        model_path : str
            Path to RWKV model file in ggml format.
        thread_count : int
            Thread count to use. If not set, defaults to CPU count / 2.
        """

        assert os.path.isfile(model_path), f'{model_path} is not a file'
        assert thread_count > 0, 'Thread count must be positive'

        self._library = shared_library

        self._ctx = self._library.rwkv_init_from_file(model_path, thread_count)

        self._state_buffer_element_count = self._library.rwkv_get_state_buffer_element_count(self._ctx)
        self._logits_buffer_element_count = self._library.rwkv_get_logits_buffer_element_count(self._ctx)

        self._valid = True

    def eval(
            self,
            token: int,
            state_in: Optional[np.ndarray],
            state_out: Optional[np.ndarray] = None,
            logits_out: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the model for a single token.
        In case of any error, this method will throw an exception.

        Parameters
        ----------
        token : int
            Index of next token to be seen by the model. Must be in range 0 <= token < n_vocab.
        state_in : Optional[np.ndarray]
            State from previous call of this method. If this is a first pass, set it to None.
        state_out : Optional[np.ndarray]
            Optional output tensor for state. If provided, must be of type float32, contiguous and of shape (state_buffer_element_count).
        logits_out : Optional[np.ndarray]
            Optional output tensor for logits. If provided, must be of type float32, contiguous and of shape (logits_buffer_element_count).

        Returns
        -------
        logits, state
            Logits vector of shape (n_vocab); state for the next step.
        """

        assert self._valid, 'Model was freed'

        def validate_buffer(buf: np.ndarray, name: str, size: int) -> None:
            assert buf.dtype == np.float32, f'{name} is not of type float32'
            assert buf.shape == (size,), f'{name} has invalid shape {buf.shape}, expected ({size})'

        if state_in is not None:
            validate_buffer(state_in, 'state_in', self._state_buffer_element_count)

            state_in_ptr = state_in.ctypes.data
        else:
            state_in_ptr = 0

        if state_out is not None:
            validate_buffer(state_out, 'state_out', self._state_buffer_element_count)
        else:
            state_out = np.zeros(self._state_buffer_element_count, dtype=np.float32)

        if logits_out is not None:
            validate_buffer(logits_out, 'logits_out', self._logits_buffer_element_count)
        else:
            logits_out = np.zeros(self._logits_buffer_element_count, dtype=np.float32)

        self._library.rwkv_eval(
            self._ctx,
            token,
            state_in_ptr,
            state_out.ctypes.data,
            logits_out.ctypes.data,
        )

        return logits_out, state_out

    def free(self):
        """
        Frees all allocated resources.
        In case of any error, this method will throw an exception.
        The object must not be used anymore after calling this method.
        """

        assert self._valid, 'Already freed'

        self._valid = False

        self._library.rwkv_free(self._ctx)

    def __del__(self):
        # Free the context on GC in case user forgot to call free() explicitly.
        if hasattr(self, '_valid') and self._valid:
            self.free()
