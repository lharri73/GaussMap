import numpy as np
cimport numpy as np

cdef extern from "include/header.hpp"
    cdef cppclass C_gaussMap "gaussMap"
    C_gaussMap()
    int* Array() const
    size_t Width() const
    size_t Height() const

cdef class gaussMap
    cdef C_gaussMap *gm

    def __cinit__(self):
        self.g = new C_gaussMap()

    def Array(self):
        return self.gm.Array()
    
    def Width(self):
        return self.gm.Width()

    def Height(self):
        return self.gm.Height()
