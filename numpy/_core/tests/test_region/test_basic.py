import unittest
from regions import Region, is_local
from immutable import freeze, isfrozen
import numpy as np

class TestBuildWithArray(unittest.TestCase):
    def setUp(self):
        class A: pass
        freeze(A())
        self.A = A
    def test_array_creation(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc+3)
    def test_array_assignment(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        c = self.A()

        original_lrc = r._lrc
        arr = np.empty(3, dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        arr[0] = r.a
        self.assertEqual(r._lrc, original_lrc+1)
        arr[1] = r.b
        self.assertEqual(r._lrc, original_lrc+2)
        arr[2] = c
        self.assertEqual(r._lrc, original_lrc+2)

if __name__ == "__main__":
    unittest.main()