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
        arr = None
        self.assertEqual(r._lrc, original_lrc)
    
    def test_array_creation_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        original_lrc = r._lrc
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_creation_region_2(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc+3)
        r.arr = arr
        self.assertEqual(r._lrc, original_lrc)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc)

    @unittest.skip("This test currently fails because the array's reference to the region's objects is not being tracked. This is a known issue that needs to be addressed.")
    def test_array_creation_with_external_ref(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc+3)
        external_ref = arr[0]
        self.assertEqual(r._lrc, original_lrc+4)
        arr = None
        external_ref = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_assignment(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        c = self.A()

        original_lrc = r._lrc
        self.assertEqual(r._lrc, original_lrc)
        arr = np.empty(3, dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        arr[0] = r.a
        self.assertEqual(r._lrc, original_lrc+1)
        arr[1] = r.b
        self.assertEqual(r._lrc, original_lrc+2)
        arr[2] = c
        self.assertEqual(r._lrc, original_lrc+2)
        arr = None
        self.assertEqual(r._lrc, original_lrc)
    
    def test_array_assignment_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        c = self.A()

        original_lrc = r._lrc
        r.arr = np.empty(3, dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        r.arr[0] = r.a
        self.assertEqual(r._lrc, original_lrc)
        r.arr[1] = r.b
        self.assertEqual(r._lrc, original_lrc)
        r.arr[2] = c
        self.assertEqual(r._lrc, original_lrc+1)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc+1) # SINCE c is still referenced by the region
        c = None
        self.assertEqual(r._lrc, original_lrc)


if __name__ == "__main__":
    unittest.main()