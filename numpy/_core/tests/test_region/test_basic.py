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
        self.assertEqual(r._lrc, original_lrc+1) # arr still points into the region
        r.arr = None
        self.assertEqual(r._lrc, original_lrc+1) # arr still points into the region
        arr = None
        self.assertEqual(r._lrc, original_lrc) # arr is freed.

    # @unittest.skip("This test currently fails because the array's reference to the region's objects is not being tracked. This is a known issue that needs to be addressed.")
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

    def test_array_inregionarr_to_localobj(self):
        r = Region()
        a = self.A()
        b = self.A()
        c = self.A()

        original_lrc = r._lrc
        r.arr = np.array([a], dtype=object)
        r.arr2 = np.array([b], dtype=object)
        r.arr3 = np.array([c], dtype=object)
        self.assertEqual(r._lrc, original_lrc+3)
        a = None
        self.assertEqual(r._lrc, original_lrc+2)
        b = None
        self.assertEqual(r._lrc, original_lrc+1)
        c = None
        self.assertEqual(r._lrc, original_lrc)

class TestBuildWithArray2(unittest.TestCase):
    def test_array_creation_region(self):
        class A: pass
        freeze(A())

        r = Region()
        r.a = A()
        r.b = A()
        r.c = A()

        original_lrc = r._lrc
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc)

class TestRegionNumpyView(unittest.TestCase):
    """Tests for numpy array view creation and LRC behavior with regions."""

    def setUp(self):
        class A: pass
        freeze(A())
        self.A = A

    def test_view_does_not_change_lrc(self):
        """
        Creating a view of a numpy array should not change the LRC,
        since a view shares the buffer and does not borrow new references
        from the region directly.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        base_lrc = r._lrc

        view = arr[1:4]
        self.assertEqual(r._lrc, base_lrc)

    def test_view_set_to_none_does_not_change_lrc(self):
        """
        Setting a view to None should not change the LRC, since
        the view does not independently borrow from the region.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        view = arr[1:4]
        base_lrc = r._lrc

        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_array_none_after_view_none_releases_lrc(self):
        """
        After both view and array are set to None, LRC should return
        to the pre-array level.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        base_lrc = r._lrc

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        view = arr[1:4]

        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_array_none_before_view_none_keeps_lrc(self):
        """
        Setting arr to None while view still exists should not release
        the buffer (view holds arr alive via base), so LRC should
        remain elevated until the view is also set to None.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        base_lrc = r._lrc

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        view = arr[1:4]

        arr = None
        # view keeps the buffer (and thus the region borrows) alive
        self.assertEqual(r._lrc, base_lrc + 6)

        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_full_slice_view_does_not_change_lrc(self):
        """
        A full slice view (arr[:]) should behave the same as a partial
        slice — no additional LRC change.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        base_lrc = r._lrc

        arr = np.array([r.a, r.b, r.c], dtype=object)
        lrc_after_array = r._lrc

        view = arr[:]
        self.assertEqual(r._lrc, lrc_after_array)

        view = None
        self.assertEqual(r._lrc, lrc_after_array)

        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_single_element_view_does_not_change_lrc(self):
        """
        A single-element slice view should not independently affect
        the LRC.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        view = arr[0:1]
        self.assertEqual(r._lrc, base_lrc)

        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_view_moved_into_region_adjusts_lrc(self):
        """
        Moving a view into a region should transfer ownership of the view
        object itself, adjusting the LRC by 1 for the external reference.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        view = arr[1:4]
        base_lrc = r._lrc  # arr borrows 6 elements

        r.view = view
        # view is now owned by region, but `view` local var still holds external ref
        self.assertEqual(r._lrc, base_lrc - 6 + 2)

        view = None
        self.assertEqual(r._lrc, base_lrc - 6 + 2 - 1)
    
    def test_view_moved_into_region_adjusts_lrc_several_views(self):
        """
        Moving multiple views into a region should adjust LRC for each view
        object, but not stack LRC for the shared base array.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()

        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        base_lrc = r._lrc  # arr borrows 6 elements

        view1 = arr[0:5]
        self.assertEqual(r._lrc, base_lrc)
        r.view1 = view1
        self.assertEqual(r._lrc, base_lrc - 6 + 2)

        r.view2 = arr[1:4]
        self.assertEqual(r._lrc, base_lrc - 6 + 2)

        view3 = arr[2:6]
        self.assertEqual(r._lrc, base_lrc - 6 + 3)

        view3 = None
        self.assertEqual(r._lrc, base_lrc - 6 + 2)

        view1 = None
        self.assertEqual(r._lrc, base_lrc - 6 + 1)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 6)

        

    def test_multiple_views_do_not_stack_lrc(self):
        """
        Creating multiple views of the same array should not
        cumulatively increase the LRC beyond what the array itself holds.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()

        arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc

        view1 = arr[0:2]
        view2 = arr[2:4]
        view3 = arr[1:3]
        self.assertEqual(r._lrc, base_lrc)

        view1 = None
        view2 = None
        view3 = None
        self.assertEqual(r._lrc, base_lrc)


class TestRegionNumpyViewWithArrayInRegion(unittest.TestCase):
    """Tests for views of arrays already inside a region."""

    def setUp(self):
        class A: pass
        freeze(A())
        self.A = A

    def test_view_of_region_array_increases_lrc(self):
        """
        Taking a view of a numpy array that is already inside a region
        should increase the LRC since the view is an external local object
        pointing into the region's array.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        view = r.arr[1:]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_view_of_region_array_released_decreases_lrc(self):
        """
        Releasing a view of a region-owned array should bring
        LRC back to pre-view level.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        view = r.arr[1:]
        self.assertEqual(r._lrc, base_lrc + 1)

        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_view_of_region_array_moved_into_region_does_not_change_lrc(self):
        """
        Moving a view of a region-owned array into the same region
        should not change LRC (both the view and its base array are owned).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        view = r.arr[1:]
        r.view = view
        view = None
        self.assertEqual(r._lrc, base_lrc)
    
    def test_view_of_region_array_released_decreases_lrc_several_views(self):
        """
        Releasing multiple views of a region-owned array should decrease
        LRC back to pre-view level.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc

        view1 = r.arr[0:2]
        view2 = r.arr[2:4]
        self.assertEqual(r._lrc, base_lrc + 2)

        view1 = None
        self.assertEqual(r._lrc, base_lrc + 1)

        view2 = None
        self.assertEqual(r._lrc, base_lrc)

if __name__ == "__main__":
    unittest.main()