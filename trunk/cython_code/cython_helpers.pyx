print("Using Cython Extensions")
import cython
import numpy as np
cimport numpy as np

DTYPE = np.complex64

ctypedef np.complex64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def estimate_states(np.ndarray[np.complex64_t,ndim=2] AKC,  np.ndarray[np.complex64_t,ndim=2] K_0m):
    
    assert AKC.dtype == np.complex64 and K_0m.dtype == np.complex64
    
    cdef int N = K_0m.shape[1]
    cdef int r = K_0m.shape[0]
    cdef np.ndarray[np.complex64_t, ndim=2] states = np.zeros([r, N], dtype=np.complex64)
    cdef np.ndarray[np.complex64_t, ndim=1] this_state = np.zeros([r], dtype=np.complex64)  
    cdef Py_ssize_t k
    cdef dot = np.dot
    
    for k in range(N-1):
        states[:,k+1] = K_0m[:,k] + dot(AKC, states[:,k])
    return states