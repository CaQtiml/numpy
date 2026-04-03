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
    
    def test_array_creation_region_violate_region(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r2.b = self.A()

        with self.assertRaises(Exception):
            r1.arr = np.array([r1.a, r2.b], dtype=object)

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
        r2 = Region()
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

# ------------- Test array_subscript ----------------------------------------------

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
        r2 = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        with self.assertRaises(Exception):
            r2.view = r.arr[1:4]

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

# ------------- Test array_ass_subscript ---------------------------------

class TestArraySubscriptAssignment(unittest.TestCase):
    """Tests for numpy array element assignment and LRC tracking."""

    def setUp(self):
        class A: pass
        freeze(A())
        self.A = A

    # ------------------------------------------------------------------
    # Basic overwrite: region element replaced by another region element
    # ------------------------------------------------------------------

    def test_overwrite_region_element_with_same_region(self):
        """
        Overwriting arr[i] with another object from the same region should
        release the borrow on the old element and acquire one on the new one.
        Net LRC change: 0.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()

        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc  # arr borrows 3

        arr[0] = r.d  # releases borrow on r.a, acquires on r.d
        self.assertEqual(r._lrc, base_lrc)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_overwrite_region_element_with_local_object(self):
        """
        Replacing a region-borrowed element with a local object should
        decrease LRC by 1 (borrow on old element released, no new borrow).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc  # borrows 3

        local_obj = self.A()
        arr[1] = local_obj  # releases borrow on r.b
        self.assertEqual(r._lrc, base_lrc - 1)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_overwrite_local_element_with_region_object(self):
        """
        Replacing a local element with a region object should increase
        LRC by 1 (new borrow acquired).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()

        local1 = self.A()
        local2 = self.A()
        arr = np.array([local1, local2], dtype=object)
        base_lrc = r._lrc  # 0 borrows from r

        arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)

        arr[1] = r.b
        self.assertEqual(r._lrc, base_lrc + 2)

        arr = None
        self.assertEqual(r._lrc, base_lrc)

    # ------------------------------------------------------------------
    # Cross-region assignment
    # ------------------------------------------------------------------

    def test_cross_region_subscript_assignment_updates_both_lrcs(self):
        """
        arr[i] = r2.x: the borrow on r (old element) is released,
        and a new borrow on r2 is acquired. Both LRCs shift accordingly.
        """
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r2.x = self.A()
        r2.y = self.A()

        arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        base_lrc1 = r1._lrc  # borrows 3
        base_lrc2 = r2._lrc  # borrows 0

        arr[0] = r2.x
        self.assertEqual(r1._lrc, base_lrc1 - 1)
        self.assertEqual(r2._lrc, base_lrc2 + 1)

        arr[1] = r2.y
        self.assertEqual(r1._lrc, base_lrc1 - 2)
        self.assertEqual(r2._lrc, base_lrc2 + 2)

        arr = None
        self.assertEqual(r1._lrc, base_lrc1 - 3)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_cross_region_full_replacement(self):
        """
        Replace all elements of an array (originally from r1) with elements
        from r2. After replacement r1 LRC should return to pre-array level.
        """
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()

        arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc

        arr[0] = r2.x
        arr[1] = r2.y
        self.assertEqual(r1._lrc, base_lrc1 - 2)
        self.assertEqual(r2._lrc, base_lrc2 + 2)

        arr = None
        self.assertEqual(r1._lrc, base_lrc1 - 2)
        self.assertEqual(r2._lrc, base_lrc2)

    # ------------------------------------------------------------------
    # Repeated overwrites of the same slot
    # ------------------------------------------------------------------

    def test_repeated_overwrite_same_slot_same_region(self):
        """
        Repeatedly overwriting arr[0] with different objects from the same
        region should keep LRC stable (each write releases old, acquires new).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()

        arr = np.array([r.a], dtype=object)
        base_lrc = r._lrc  # borrows 1

        arr[0] = r.b
        self.assertEqual(r._lrc, base_lrc)
        arr[0] = r.c
        self.assertEqual(r._lrc, base_lrc)
        arr[0] = r.d
        self.assertEqual(r._lrc, base_lrc)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 1)

    def test_repeated_overwrite_same_slot_with_local_then_region(self):
        """
        Slot starts with region object, replaced by local (LRC -1),
        then replaced by another region object (LRC back to base).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()

        arr = np.array([r.a], dtype=object)
        base_lrc = r._lrc

        local = self.A()
        arr[0] = local
        self.assertEqual(r._lrc, base_lrc - 1)

        arr[0] = r.b
        self.assertEqual(r._lrc, base_lrc)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 1)

    # ------------------------------------------------------------------
    # Region-owned array subscript assignment
    # ------------------------------------------------------------------

    def test_region_array_overwrite_with_local_increases_lrc(self):
        """
        r.arr[i] = local_obj: the array is inside the region.
        Assigning a local object into a slot previously holding a region object
        should increase the external-reference count by 1 (local now referenced
        from inside the region).
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc

        local = self.A()
        r.arr[0] = local  # region now holds a reference to local
        self.assertEqual(r._lrc, base_lrc + 1)

        r.arr[0] = r.a  
        self.assertEqual(r._lrc, base_lrc + 1)  # "local" still points to the object in the region.

    def test_region_array_overwrite_local_slot_with_another_local(self):
        """
        Both old and new values for a slot are local objects.
        Replacing one local with another should keep LRC stable.
        """
        r = Region()
        local1 = self.A()
        local2 = self.A()
        local3 = self.A()
        r.arr = np.array([local1, local2], dtype=object)
        base_lrc = r._lrc  # borrows 2 locals

        r.arr[0] = local3  # releases local1, acquires local3 — net 0
        self.assertEqual(r._lrc, base_lrc+1)

    def test_region_array_overwrite_local_slot_with_region_object(self):
        """
        r.arr[i] was holding a local reference (LRC +1).
        Overwriting with a region-owned object releases that borrow (LRC -1).
        """
        r = Region()
        r.a = self.A()
        local = self.A()
        r.arr = np.array([local], dtype=object)
        base_lrc = r._lrc  # +1 for local held inside region

        r.arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc)

    # ------------------------------------------------------------------
    # Cross-region assignment into region-owned array should raise
    # ------------------------------------------------------------------

    def test_region_array_cross_region_assign_raises(self):
        """
        Assigning an object from r2 into r1's array should raise an exception
        (region isolation violation).
        """
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        r2.b = self.A()

        with self.assertRaises(Exception):
            r1.arr[0] = r2.b

    # ------------------------------------------------------------------
    # Slice assignment
    # ------------------------------------------------------------------

    def test_slice_assignment_from_same_region(self):
        """
        arr[0:2] = [r.c, r.d] replaces two region borrows with two other
        region borrows — net LRC unchanged.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()

        arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc

        arr[0:2] = np.array([r.c, r.d], dtype=object)
        self.assertEqual(r._lrc, base_lrc)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 2)

    def test_slice_assignment_replaces_region_with_locals(self):
        """
        Slice-assigning local objects over region-borrowed slots should
        decrease LRC by the number of replaced region objects.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()

        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc  # borrows 3

        locals_ = np.array([self.A(), self.A(), self.A()], dtype=object)
        arr[0:3] = locals_
        self.assertEqual(r._lrc, base_lrc - 3)

        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_slice_assignment_replaces_locals_with_region(self):
        """
        Slice-assigning region objects over local slots should increase LRC.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()

        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc

        arr[0:2] = np.array([r.a, r.b], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)

        arr = None
        self.assertEqual(r._lrc, base_lrc)
    
    def test_slice_assignment_replaces_region_with_local(self):
        """
        Slice-assigning a local object over a region-borrowed slot should
        decrease LRC by 1.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()

        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc

        local1 = self.A()
        local2 = self.A()
        r.arr[0:2] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc+2)

        local1 = None
        self.assertEqual(r._lrc, base_lrc+1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    # ------------------------------------------------------------------
    # Assignment into empty / None-initialised array
    # ------------------------------------------------------------------

    def test_assign_into_zeros_array(self):
        """
        np.empty initialises slots to None-equivalent. Assigning region
        objects into those slots should increase LRC normally.
        """
        r = Region()
        r.a = self.A()
        r.b = self.A()
        base_lrc = r._lrc

        arr = np.empty(2, dtype=object)
        arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)
        arr[1] = r.b
        self.assertEqual(r._lrc, base_lrc + 2)

        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_assign_then_clear_slot_to_local(self):
        """
        After assigning a region object to a slot, overwriting with None
        (or a local) should release that borrow.
        """
        r = Region()
        r.a = self.A()
        base_lrc = r._lrc

        arr = np.empty(1, dtype=object)
        arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)

        arr[0] = None  # treat as local / no-region reference
        self.assertEqual(r._lrc, base_lrc)

    # ------------------------------------------------------------------
    # LRC consistency after del on array element (if supported)
    # ------------------------------------------------------------------

    def test_lrc_stable_across_multiple_cross_region_swaps(self):
        """
        Alternating assignments between r1 and r2 objects in the same slot
        should keep each region's LRC consistent throughout.
        """
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r2.b = self.A()

        arr = np.array([r1.a], dtype=object)
        base_lrc1 = r1._lrc  # 1 borrow
        base_lrc2 = r2._lrc  # 0 borrows

        arr[0] = r2.b
        self.assertEqual(r1._lrc, base_lrc1 - 1)
        self.assertEqual(r2._lrc, base_lrc2 + 1)

        arr[0] = r1.a
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

        arr[0] = r2.b
        self.assertEqual(r1._lrc, base_lrc1 - 1)
        self.assertEqual(r2._lrc, base_lrc2 + 1)

        arr = None
        self.assertEqual(r1._lrc, base_lrc1 - 1)
        self.assertEqual(r2._lrc, base_lrc2)

if __name__ == "__main__":
    unittest.main()