import ctypes
import numpy as np
import unittest
import os

def find_openblas():
    lib_path = r"C:\Users\User\Downloads\OpenBLAS-0.3.31-x64\win64\bin\libopenblas.dll"
    #lib_path = "/mnt/c/Users/User/Downloads/OpenBLAS-0.3.31-x64/win64/bin/libopenblas.dll"

    if os.path.exists(lib_path):
        return lib_path

    possible_paths = [
        "/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so",
        "libopenblas.dll",
        "libopenblas.so",
        "libopenblas.dylib",
        "/usr/lib/x86_64-linux-gnu/libopenblas.so",
        "/usr/local/lib/libopenblas.dylib"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    return None


LIB_PATH = find_openblas()
if LIB_PATH is None:
    raise RuntimeError("OpenBLAS library not found!")

lib = ctypes.CDLL(LIB_PATH)

# Константы CBLAS
CblasRowMajor = 101
CblasColMajor = 102
CblasNoTrans = 111
CblasTrans = 112
CblasConjTrans = 113
CblasUpper = 121
CblasLower = 122
CblasNonUnit = 131
CblasUnit = 132

lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
lib.openblas_set_num_threads.restype = None

def setup_argtypes():
    # GEMV
    lib.cblas_sgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_float,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_double, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_double,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_cgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_zgemv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int]

    # TRMV
    lib.cblas_strmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dtrmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ctrmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ztrmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    # SYMV
    lib.cblas_ssymv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dsymv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_double, ctypes.c_void_p, ctypes.c_int]

    # GER
    lib.cblas_sger.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                               ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dger.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                               ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_cgeru.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_cgerc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_zgeru.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_zgerc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int]

    # TRSV
    lib.cblas_strsv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dtrsv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ctrsv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ztrsv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_int]

    # TPMV
    lib.cblas_stpmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_dtpmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ctpmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

    lib.cblas_ztpmv.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


setup_argtypes()

for name in dir(lib):
    if name.startswith('cblas_'):
        getattr(lib, name).restype = None

EPS_F = 1e-6
EPS_D = 1e-12

class TestCBLASLevel2(unittest.TestCase):

    def setUp(self):
        lib.openblas_set_num_threads(1)

    #GEMV тесты
    def test_sgemv(self):
        A = np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        expected = np.array([3, 4], dtype=np.float32)

        lib.cblas_sgemv(
            CblasRowMajor, CblasNoTrans, 2, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)

    def test_dgemv(self):
        A = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float64)
        y = np.zeros(2, dtype=np.float64)
        expected = np.array([3, 7], dtype=np.float64)

        lib.cblas_dgemv(
            CblasRowMajor, CblasNoTrans, 2, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_D, atol=EPS_D)

    def test_cgemv(self):
        A = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64).reshape(2, 2)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex64)
        y = np.zeros(2, dtype=np.complex64)
        alpha = np.array([1 + 0j], dtype=np.complex64)
        beta = np.array([0 + 0j], dtype=np.complex64)
        expected = np.array([3 + 3j, 7 + 7j], dtype=np.complex64)

        lib.cblas_cgemv(
            CblasRowMajor, CblasNoTrans, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            beta.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)

    def test_zgemv(self):
        A = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128).reshape(2, 2)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex128)
        y = np.zeros(2, dtype=np.complex128)
        alpha = np.array([1 + 0j], dtype=np.complex128)
        beta = np.array([0 + 0j], dtype=np.complex128)
        expected = np.array([3 + 3j, 7 + 7j], dtype=np.complex128)

        lib.cblas_zgemv(
            CblasRowMajor, CblasNoTrans, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            beta.ctypes.data_as(ctypes.c_void_p),
            y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_D, atol=EPS_D)

    #TRMV тесты
    def test_strmv(self):
        A = np.array([1, 2, 0, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float32)
        expected = np.array([3, 4], dtype=np.float32)

        lib.cblas_strmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_dtrmv(self):
        A = np.array([1, 2, 0, 4], dtype=np.float64).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float64)
        expected = np.array([3, 4], dtype=np.float64)

        lib.cblas_dtrmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    def test_ctrmv(self):
        A = np.array([1 + 0j, 2 + 0j, 0 + 0j, 4 + 0j], dtype=np.complex64).reshape(2, 2)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex64)
        expected = np.array([3 + 0j, 4 + 0j], dtype=np.complex64)

        lib.cblas_ctrmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_ztrmv(self):
        A = np.array([1 + 0j, 2 + 0j, 0 + 0j, 4 + 0j], dtype=np.complex128).reshape(2, 2)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex128)
        expected = np.array([3 + 0j, 4 + 0j], dtype=np.complex128)

        lib.cblas_ztrmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    #SYMV тесты
    def test_ssymv(self):
        A = np.array([1, 2, 2, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        expected = np.array([3, 6], dtype=np.float32)

        lib.cblas_ssymv(
            CblasRowMajor, CblasUpper, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)

    def test_dsymv(self):
        A = np.array([1, 2, 2, 4], dtype=np.float64).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float64)
        y = np.zeros(2, dtype=np.float64)
        expected = np.array([3, 6], dtype=np.float64)

        lib.cblas_dsymv(
            CblasRowMajor, CblasUpper, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_D, atol=EPS_D)

    #GER тесты
    def test_sger(self):
        A = np.zeros(4, dtype=np.float32).reshape(2, 2)
        x = np.array([2, 3], dtype=np.float32)
        y = np.array([4, 5], dtype=np.float32)
        expected = np.array([8, 10, 12, 15], dtype=np.float32).reshape(2, 2)

        lib.cblas_sger(
            CblasRowMajor, 2, 2, 1.0,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )

        np.testing.assert_allclose(A, expected, rtol=EPS_F, atol=EPS_F)

    def test_dger(self):
        A = np.zeros(4, dtype=np.float64).reshape(2, 2)
        x = np.array([2, 3], dtype=np.float64)
        y = np.array([4, 5], dtype=np.float64)
        expected = np.array([8, 10, 12, 15], dtype=np.float64).reshape(2, 2)

        lib.cblas_dger(
            CblasRowMajor, 2, 2, 1.0,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )

        np.testing.assert_allclose(A, expected, rtol=EPS_D, atol=EPS_D)

    def test_cgerc(self):
        A = np.zeros(4, dtype=np.complex64).reshape(2, 2)
        x = np.array([2 + 1j, 3 + 2j], dtype=np.complex64)
        y = np.array([4 + 3j, 5 + 4j], dtype=np.complex64)
        alpha = np.array([1 + 0j], dtype=np.complex64)

        lib.cblas_cgerc(
            CblasRowMajor, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )


        self.assertTrue(np.any(A != 0))

    def test_zgerc(self):
        A = np.zeros(4, dtype=np.complex128).reshape(2, 2)
        x = np.array([2 + 1j, 3 + 2j], dtype=np.complex128)
        y = np.array([4 + 3j, 5 + 4j], dtype=np.complex128)
        alpha = np.array([1 + 0j], dtype=np.complex128)

        lib.cblas_zgerc(
            CblasRowMajor, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )

        self.assertTrue(np.any(A != 0))

    def test_cgeru(self):
        A = np.zeros(4, dtype=np.complex64).reshape(2, 2)
        x = np.array([2 + 1j, 3 + 2j], dtype=np.complex64)
        y = np.array([4 + 3j, 5 + 4j], dtype=np.complex64)
        alpha = np.array([1 + 0j], dtype=np.complex64)

        lib.cblas_cgeru(
            CblasRowMajor, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )

        self.assertTrue(np.any(A != 0))

    def test_zgeru(self):
        A = np.zeros(4, dtype=np.complex128).reshape(2, 2)
        x = np.array([2 + 1j, 3 + 2j], dtype=np.complex128)
        y = np.array([4 + 3j, 5 + 4j], dtype=np.complex128)
        alpha = np.array([1 + 0j], dtype=np.complex128)

        lib.cblas_zgeru(
            CblasRowMajor, 2, 2,
            alpha.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1,
            y.ctypes.data_as(ctypes.c_void_p), 1,
            A.ctypes.data_as(ctypes.c_void_p), 2
        )

        self.assertTrue(np.any(A != 0))

    #TRSV тесты
    def test_strsv(self):
        A = np.array([1, 2, 0, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([3, 4], dtype=np.float32)
        expected = np.array([1, 1], dtype=np.float32)

        lib.cblas_strsv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_dtrsv(self):
        A = np.array([1, 2, 0, 4], dtype=np.float64).reshape(2, 2)
        x = np.array([3, 4], dtype=np.float64)
        expected = np.array([1, 1], dtype=np.float64)

        lib.cblas_dtrsv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    def test_ctrsv(self):
        A = np.array([1 + 0j, 2 + 0j, 0 + 0j, 4 + 0j], dtype=np.complex64).reshape(2, 2)
        x = np.array([3 + 0j, 4 + 0j], dtype=np.complex64)
        expected = np.array([1 + 0j, 1 + 0j], dtype=np.complex64)

        lib.cblas_ctrsv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_ztrsv(self):
        A = np.array([1 + 0j, 2 + 0j, 0 + 0j, 4 + 0j], dtype=np.complex128).reshape(2, 2)
        x = np.array([3 + 0j, 4 + 0j], dtype=np.complex128)
        expected = np.array([1 + 0j, 1 + 0j], dtype=np.complex128)

        lib.cblas_ztrsv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    #TPMV тесты
    def test_stpmv(self):
        Ap = np.array([1, 2, 4], dtype=np.float32)
        x = np.array([1, 1], dtype=np.float32)
        expected = np.array([3, 4], dtype=np.float32)

        lib.cblas_stpmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            Ap.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_dtpmv(self):
        Ap = np.array([1, 2, 4], dtype=np.float64)
        x = np.array([1, 1], dtype=np.float64)
        expected = np.array([3, 4], dtype=np.float64)

        lib.cblas_dtpmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            Ap.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    def test_ctpmv(self):
        Ap = np.array([1 + 0j, 2 + 0j, 4 + 0j], dtype=np.complex64)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex64)
        expected = np.array([3 + 0j, 4 + 0j], dtype=np.complex64)

        lib.cblas_ctpmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            Ap.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_F, atol=EPS_F)

    def test_ztpmv(self):
        Ap = np.array([1 + 0j, 2 + 0j, 4 + 0j], dtype=np.complex128)
        x = np.array([1 + 0j, 1 + 0j], dtype=np.complex128)
        expected = np.array([3 + 0j, 4 + 0j], dtype=np.complex128)

        lib.cblas_ztpmv(
            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 2,
            Ap.ctypes.data_as(ctypes.c_void_p),
            x.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(x, expected, rtol=EPS_D, atol=EPS_D)

    #Дополнительные тесты
    def test_gemv_beta_nonzero(self):
        A = np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float32)
        y = np.array([1, 1], dtype=np.float32)
        expected = A @ x + y  # alpha=1, beta=1

        lib.cblas_sgemv(
            CblasRowMajor, CblasNoTrans, 2, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            1.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)

    def test_gemv_negative_alpha(self):
        A = np.array([1, 2, 3, 4], dtype=np.float32).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float32)
        y = np.zeros(2, dtype=np.float32)
        expected = -2.0 * (A @ x)

        lib.cblas_sgemv(
            CblasRowMajor, CblasNoTrans, 2, 2, -2.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_F, atol=EPS_F)

    def test_multithreaded(self):
        lib.openblas_set_num_threads(4)

        A = np.array([1, 2, 3, 4], dtype=np.float64).reshape(2, 2)
        x = np.array([1, 1], dtype=np.float64)
        y = np.zeros(2, dtype=np.float64)
        expected = np.array([3, 7], dtype=np.float64)

        lib.cblas_dgemv(
            CblasRowMajor, CblasNoTrans, 2, 2, 1.0,
            A.ctypes.data_as(ctypes.c_void_p), 2,
            x.ctypes.data_as(ctypes.c_void_p), 1,
            0.0, y.ctypes.data_as(ctypes.c_void_p), 1
        )

        np.testing.assert_allclose(y, expected, rtol=EPS_D, atol=EPS_D)

        lib.openblas_set_num_threads(1)


if __name__ == '__main__':
    print("CBLAS LEVEL 2 TESTS (Python port)")
    print(f"Using OpenBLAS from: {LIB_PATH}")
    print()

    unittest.main(argv=[''], verbosity=2)