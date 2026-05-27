import unittest
from regions import Region, is_local
from immutable import freeze, set_freezable, FREEZABLE_YES
import numpy as np


# ===========================================================================
# Helpers
# ===========================================================================

def make_A():
    class A: pass
    freeze(A())
    return A


# ===========================================================================
# Array creation and ownership
# ===========================================================================

class TestArrayCreation(unittest.TestCase):
    """
    Tests for numpy object-array creation and region ownership.
    Covers LRC accounting when arrays are constructed from region objects,
    local objects, or a mix, and when ownership is transferred into a region.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    def test_array_creation_local_borrows(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc + 3)
        arr = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_creation_region_owned_no_lrc_change(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        original_lrc = r._lrc
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_creation_cross_region_violates(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r2.b = self.A()
        with self.assertRaises(Exception):
            r1.arr = np.array([r1.a, r2.b], dtype=object)

    def test_array_creation_local_then_moved_to_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc + 3)
        r.arr = arr
        self.assertEqual(r._lrc, original_lrc + 1)
        r.arr = None
        self.assertEqual(r._lrc, original_lrc + 1)
        arr = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_creation_with_external_ref(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        original_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, original_lrc + 3)
        external_ref = arr[0]
        self.assertEqual(r._lrc, original_lrc + 4)
        arr = None
        external_ref = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_assignment_local_arr_mixed(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        c = self.A()
        original_lrc = r._lrc
        arr = np.empty(3, dtype=object)
        self.assertEqual(r._lrc, original_lrc)
        arr[0] = r.a
        self.assertEqual(r._lrc, original_lrc + 1)
        arr[1] = r.b
        self.assertEqual(r._lrc, original_lrc + 2)
        arr[2] = c
        self.assertEqual(r._lrc, original_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, original_lrc)

    def test_array_assignment_region_arr_mixed(self):
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
        self.assertEqual(r._lrc, original_lrc + 1)
        self.assertTrue(r.owns(c))
        r.arr = None
        self.assertEqual(r._lrc, original_lrc + 1)
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
        self.assertEqual(r._lrc, original_lrc + 3)
        a = None
        self.assertEqual(r._lrc, original_lrc + 2)
        b = None
        self.assertEqual(r._lrc, original_lrc + 1)
        c = None
        self.assertEqual(r._lrc, original_lrc)


# ===========================================================================
# array_subscript — HAS_SLICE
# ===========================================================================

class TestArraySubscript_Slice(unittest.TestCase):
    """
    Tests for array_subscript via HAS_SLICE.
    arr[i:j] returns a view sharing the base buffer — no independent borrow
    is created for the view itself when the source array is local. When the
    source array is region-owned, the view is a local external reference and
    LRC increases by 1.
    Covers view lifetime, buffer ownership, and cross-region isolation.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_view_of_local_arr_does_not_change_lrc(self):
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

    def test_view_release_does_not_change_lrc(self):
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

    def test_full_slice_view_does_not_change_lrc(self):
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

    def test_multiple_views_do_not_stack_lrc(self):
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

    # Guideline 2 — local source, regional target (adding reference)
    def test_arr_none_before_view_keeps_lrc(self):
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
        self.assertEqual(r._lrc, base_lrc + 6)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 3 — local source, regional target (removing reference)
    def test_arr_none_after_view_releases_lrc(self):
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

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_view_of_region_array_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[1:]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_view_of_region_array_released_decreases_lrc(self):
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

    def test_view_of_region_array_moved_into_same_region_no_lrc_change(self):
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

    def test_view_of_region_array_several_views_decreases_one_by_one(self):
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

    def test_view_moved_into_region_adjusts_lrc(self):
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
        r.view = view
        self.assertEqual(r._lrc, base_lrc - 6 + 2)
        view = None
        self.assertEqual(r._lrc, base_lrc - 6 + 2 - 1)

    def test_view_moved_into_region_adjusts_lrc_region_owned_base(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        r.arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        base_lrc = r._lrc
        view = r.arr[1:4]
        self.assertEqual(r._lrc, base_lrc + 1)
        r.view = view
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_several_views_moved_into_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        arr = np.array([r.a, r.b, r.c, r.d, r.e, r.f], dtype=object)
        base_lrc = r._lrc
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

    # Guideline 5 — cross-region raises
    def test_view_of_region_array_into_other_region_raises(self):
        r = Region()
        r2 = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        with self.assertRaises(Exception):
            r2.view = r.arr[1:4]

    def test_slice_stored_in_other_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r1.d = self.A()
        r1.e = self.A()
        r1.arr = np.array([r1.a, r1.b, r1.c, r1.d, r1.e], dtype=object)
        with self.assertRaises(Exception):
            r2.stolen = r1.arr[1:3]


# ===========================================================================
# array_subscript — HAS_ELLIPSIS
# ===========================================================================

class TestArraySubscript_Ellipsis(unittest.TestCase):
    """
    Tests for array_subscript via HAS_ELLIPSIS.
    arr[...] returns a view of self sharing the same buffer. When the source
    is a local array, the ellipsis view carries no independent borrow. When
    the source is region-owned, the view is a local external reference (LRC +1).
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_ellipsis_of_local_arr_does_not_change_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_view_release_does_not_change_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        view = arr[...]
        base_lrc = r._lrc
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_view_both_none_releases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        base_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        view = arr[...]
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_ellipsis_arr_none_before_view_keeps_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        base_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        view = arr[...]
        arr = None
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_ellipsis_of_region_array_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_ellipsis_of_region_array_released_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_of_region_array_moved_into_same_region_no_lrc_change(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[...]
        r.view = view
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_multiple_ellipsis_views_lrc_additive(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        view1 = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 1)
        view2 = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 2)
        view1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        view2 = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_ellipsis_of_region_array_into_other_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        with self.assertRaises(Exception):
            r2.view = r1.arr[...]


# ===========================================================================
# array_subscript — HAS_INTEGER
# ===========================================================================

class TestArraySubscript_Integer(unittest.TestCase):
    """
    Tests for array_subscript via HAS_INTEGER.
    arr[i] on a 1-d object array returns the object directly, creating a new
    local reference. LRC increases by 1 per extraction from a region-owned
    slot, and decreases when that reference is released.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_scalar_get_from_local_element_does_not_change_lrc(self):
        r = Region()
        r.a = self.A()
        local = self.A()
        arr = np.array([r.a, local], dtype=object)
        base_lrc = r._lrc
        item = arr[1]
        self.assertEqual(r._lrc, base_lrc)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    def test_scalar_get_arr_released_before_item_local(self):
        r = Region()
        a = self.A()
        b = self.A()
        c = self.A()
        base_lrc = r._lrc
        arr = np.array([a, b, c], dtype=object)
        item = arr[1]
        arr = None
        self.assertEqual(r._lrc, base_lrc)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_scalar_get_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item = arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_multiple_scalar_gets_accumulate_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item0 = arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)
        item1 = arr[1]
        self.assertEqual(r._lrc, base_lrc + 2)
        item2 = arr[2]
        self.assertEqual(r._lrc, base_lrc + 3)
        item0 = None
        self.assertEqual(r._lrc, base_lrc + 2)
        item1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        item2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_multiple_scalar_gets_mixed_local_and_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item0 = arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)
        r.item1 = arr[1]
        self.assertEqual(r._lrc, base_lrc + 1)
        item2 = arr[2]
        self.assertEqual(r._lrc, base_lrc + 2)
        item0 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        r.item1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        item2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_scalar_get_same_slot_twice_accumulates_lrc(self):
        r = Region()
        r.a = self.A()
        arr = np.array([r.a], dtype=object)
        base_lrc = r._lrc
        ref1 = arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)
        ref2 = arr[0]
        self.assertEqual(r._lrc, base_lrc + 2)
        ref1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        ref2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_scalar_get_negative_index(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item = arr[-1]
        self.assertEqual(r._lrc, base_lrc + 1)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 3 — local source, regional target (removing reference)
    def test_scalar_get_release_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item = arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    def test_scalar_get_arr_released_before_item(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        base_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        item = arr[1]
        arr = None
        self.assertEqual(r._lrc, base_lrc + 1)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_scalar_get_from_region_array_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        item = r.arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_scalar_get_from_region_array_release_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        item = r.arr[0]
        self.assertEqual(r._lrc, base_lrc + 1)
        item = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_scalar_get_into_other_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        with self.assertRaises(Exception):
            r2.stolen = r1.arr[0]


# ===========================================================================
# array_subscript — HAS_BOOL
# ===========================================================================

class TestArraySubscript_Bool(unittest.TestCase):
    """
    Tests for array_subscript via HAS_BOOL.
    arr[bool_mask] produces an independent COPY (not a view) of the selected
    elements. The copy holds its own borrows on the source region, independent
    of the source array's lifetime.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_bool_get_all_local_no_lrc_change(self):
        r = Region()
        a = self.A()
        b = self.A()
        c = self.A()
        base_lrc = r._lrc
        arr = np.array([a, b, c], dtype=object)
        mask = np.array([True, False, True], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc)
        result = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_get_local_result_is_independent_copy(self):
        r = Region()
        a = self.A()
        b = self.A()
        base_lrc = r._lrc
        arr = np.array([a, b], dtype=object)
        mask = np.array([True, True], dtype=bool)
        result = arr[mask]
        arr = None
        self.assertEqual(r._lrc, base_lrc)
        result = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_bool_get_region_elements_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc + 2)

    def test_bool_get_all_true_mask_borrows_all(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True, True], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc + 3)

    def test_bool_get_single_true_borrows_one(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([False, True, False], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_bool_get_all_false_mask_borrows_none(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([False, False, False], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc)
        result = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 3 — local source, regional target (removing reference)
    def test_bool_get_result_release_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        result = arr[mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        result = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_get_source_release_does_not_affect_result_borrows(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        base_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        mask = np.array([True, True, True], dtype=bool)
        result = arr[mask]
        arr = None
        self.assertEqual(r._lrc, base_lrc + 3)
        result = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_get_two_results_release_independently(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc
        result1 = arr[np.array([True, True, False, False], dtype=bool)]
        result2 = arr[np.array([False, False, True, True], dtype=bool)]
        self.assertEqual(r._lrc, base_lrc + 4)
        result1 = None
        self.assertEqual(r._lrc, base_lrc + 2)
        result2 = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_bool_get_from_region_array_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        result = r.arr[mask]
        self.assertEqual(r._lrc, base_lrc + 2)

    def test_bool_get_from_region_array_release_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True, False], dtype=bool)
        result = r.arr[mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        result = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_bool_get_result_into_other_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r1.arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        mask = np.array([True, True, False], dtype=bool)
        with self.assertRaises(Exception):
            r2.result = r1.arr[mask]

    def test_bool_get_cross_region_lrc_stable_after_failure(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True, True], dtype=bool)
        try:
            r2.result = r1.arr[mask]
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)


# ===========================================================================
# array_assign_subscript — LHS: direct integer (arr[i] = value)
# ===========================================================================

class TestArrayAssign_DirectInteger(unittest.TestCase):
    """
    Tests for array_assign_subscript via HAS_INTEGER on the left-hand side.
    arr[i] = value routes through PyArray_Pack_DuckTape which atomically
    releases the old slot borrow and acquires a new one.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_assign_into_empty_slot(self):
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

    def test_assign_then_clear_slot_to_none(self):
        r = Region()
        r.a = self.A()
        base_lrc = r._lrc
        arr = np.empty(1, dtype=object)
        arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)
        arr[0] = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_overwrite_local_with_region_object(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        local1 = self.A()
        local2 = self.A()
        arr = np.array([local1, local2], dtype=object)
        base_lrc = r._lrc
        arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)
        arr[1] = r.b
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_repeated_overwrite_same_slot_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        arr = np.array([r.a], dtype=object)
        base_lrc = r._lrc
        arr[0] = r.b
        self.assertEqual(r._lrc, base_lrc)
        arr[0] = r.c
        self.assertEqual(r._lrc, base_lrc)
        arr[0] = r.d
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 1)

    def test_cross_region_assignment_updates_both_lrcs(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r2.x = self.A()
        r2.y = self.A()
        arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
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

    def test_lrc_stable_across_multiple_cross_region_swaps(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r2.b = self.A()
        arr = np.array([r1.a], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
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

    # Guideline 3 — local source, regional target (removing reference)
    def test_overwrite_region_element_with_local(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_obj = self.A()
        arr[1] = local_obj
        self.assertEqual(r._lrc, base_lrc - 1)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_overwrite_region_with_same_region_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        arr[0] = r.d
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_repeated_overwrite_local_then_region(self):
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

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_region_array_overwrite_with_local_increases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local = self.A()
        r.arr[0] = local
        self.assertEqual(r._lrc, base_lrc + 1)
        self.assertTrue(r.owns(local))
        r.arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc + 1)

    def test_region_array_overwrite_local_with_another_local(self):
        r = Region()
        local1 = self.A()
        local2 = self.A()
        local3 = self.A()
        r.arr = np.array([local1, local2], dtype=object)
        base_lrc = r._lrc
        r.arr[0] = local3
        self.assertEqual(r._lrc, base_lrc + 1)
        self.assertTrue(r.owns(local3))
        self.assertTrue(r.owns(local1))
        self.assertTrue(r.owns(local2))

    def test_region_array_overwrite_local_with_region_object(self):
        r = Region()
        r.a = self.A()
        local = self.A()
        r.arr = np.array([local], dtype=object)
        base_lrc = r._lrc
        r.arr[0] = r.a
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_region_array_cross_region_assign_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        r2.b = self.A()
        with self.assertRaises(Exception):
            r1.arr[0] = r2.b
        self.assertEqual(r1.arr[0], r1.a)


# ===========================================================================
# array_assign_subscript — LHS: direct slice (arr[i:j] = value)
# ===========================================================================

class TestArrayAssign_DirectSlice(unittest.TestCase):
    """
    Tests for array_assign_subscript via HAS_SLICE on the left-hand side.
    arr[i:j] = values routes through get_view_from_index + CopyObject,
    applying per-element borrow accounting across all selected slots.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_slice_assign_all_local_no_lrc_change(self):
        r = Region()
        base_lrc = r._lrc
        arr = np.array([self.A(), self.A()], dtype=object)
        arr[0:2] = np.array([self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_slice_assign_region_over_locals(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        arr[0:2] = np.array([r.a, r.b], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_same_region_net_zero(self):
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

    # Guideline 3 — local source, regional target (removing reference)
    def test_slice_assign_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        arr[0:3] = np.array([self.A(), self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 3)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_slice_assign_local_into_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        r.arr[0:2] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_slice_assign_cross_region_into_region_array_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        with self.assertRaises(Exception):
            r1.arr[0:2] = np.array([r2.x, r2.y], dtype=object)

    def test_slice_assign_partial_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        with self.assertRaises(Exception):
            r1.arr[1:2] = np.array([r2.x], dtype=object)

    def test_slice_assign_single_element_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        r2.x = self.A()
        with self.assertRaises(Exception):
            r1.arr[0:1] = np.array([r2.x], dtype=object)

    def test_slice_assign_cross_region_lrc_stable_after_failure(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            r1.arr[0:2] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)


# ===========================================================================
# array_assign_subscript — LHS: direct ellipsis (arr[...] = value)
# ===========================================================================

class TestArrayAssign_DirectEllipsis(unittest.TestCase):
    """
    Tests for array_assign_subscript via HAS_ELLIPSIS on the left-hand side.
    arr[...] = values routes through CopyObject(self, op), applying
    per-element borrow accounting across all slots simultaneously.
    Also covers assignment through an intermediate ellipsis view variable.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 2 — local source, regional target (adding reference)
    def test_ellipsis_assign_region_over_locals(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        arr[...] = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_same_region_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        arr[...] = np.array([r.d, r.e, r.f], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    # Guideline 3 — local source, regional target (removing reference)
    def test_ellipsis_assign_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        arr[...] = np.array([self.A(), self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 3)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_ellipsis_assign_same_region_on_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.e = self.A()
        r.f = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        r.arr[...] = np.array([r.d, r.e, r.f], dtype=object)
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_locals_into_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        local3 = self.A()
        r.arr[...] = np.array([local1, local2, local3], dtype=object)
        self.assertTrue(r.owns(local1))
        self.assertTrue(r.owns(local2))
        self.assertTrue(r.owns(local3))
        self.assertEqual(r._lrc, base_lrc + 3)
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 2)
        local2 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local3 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_region_over_locals_in_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        local1 = self.A()
        local2 = self.A()
        base_lrc = r._lrc
        r.arr = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)

        base_lrc2 = r._lrc
        r.arr[...] = np.array([r.a, r.b], dtype=object)
        self.assertTrue(r.owns(local1))
        self.assertTrue(r.owns(local2))
        self.assertEqual(r._lrc, base_lrc2)

    # Guideline 5 — cross-region raises
    def test_ellipsis_assign_cross_region_into_region_array_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        with self.assertRaises(Exception):
            r1.arr[...] = np.array([r2.x, r2.y], dtype=object)

    def test_ellipsis_assign_cross_region_lrc_stable_after_failure(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            r1.arr[...] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # Through ellipsis view variable
    def test_integer_assign_through_ellipsis_view_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        self.assertEqual(r._lrc, base_lrc)
        view[0] = r.x
        self.assertEqual(r._lrc, base_lrc)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_integer_assign_local_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        local = self.A()
        view[0] = local
        self.assertEqual(r._lrc, base_lrc - 1)
        view = None
        self.assertEqual(r._lrc, base_lrc - 1)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_integer_assign_region_through_ellipsis_view_over_local(self):
        r = Region()
        r.x = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0] = r.x
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_integer_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[0] = r2.x

    def test_integer_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0] = r2.x
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_slice_assign_through_ellipsis_view_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0:2] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_slice_assign_locals_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0:2] = np.array([self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 2)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_slice_assign_region_through_ellipsis_view_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0:2] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[0:2] = np.array([r2.x, r2.y], dtype=object)

    def test_slice_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0:2] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_ellipsis_assign_through_ellipsis_view_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        r.z = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[...] = np.array([r.x, r.y, r.z], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_ellipsis_assign_locals_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[...] = np.array([self.A(), self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 3)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_ellipsis_assign_region_through_ellipsis_view_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        r.z = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[...] = np.array([r.x, r.y, r.z], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_locals_through_ellipsis_view_of_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        view = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 1)
        local1 = self.A()
        local2 = self.A()
        view[...] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        self.assertEqual(r._lrc, base_lrc + 2)
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[...] = np.array([r2.x, r2.y], dtype=object)

    def test_ellipsis_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[...] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)


# ===========================================================================
# array_assign_subscript — LHS: direct bool mask (arr[mask] = value)
# ===========================================================================

class TestArrayAssign_DirectBool(unittest.TestCase):
    """
    Tests for array_assign_subscript via HAS_BOOL on the left-hand side.
    arr[mask] = values routes through array_assign_boolean_subscript.
    Covers all four RHS index types: direct values, HAS_INTEGER, HAS_SLICE,
    and HAS_ELLIPSIS sources, as well as HAS_BOOL RHS from the four-by-four matrix.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 1 — both local
    def test_bool_assign_all_local_no_lrc_change(self):
        r = Region()
        a = self.A()
        b = self.A()
        x = self.A()
        base_lrc = r._lrc
        arr = np.array([a, b], dtype=object)
        mask = np.array([True, False], dtype=bool)
        arr[mask] = np.array([x], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_assign_all_false_mask_no_lrc_change(self):
        r = Region()
        r.x = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        mask = np.array([False, False], dtype=bool)
        arr[mask] = np.array([], dtype=object)
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 2 — local source, regional target (adding reference)
    def test_bool_assign_region_values_into_local_array(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        arr[mask] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_assign_all_true_mask_borrows_all(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True, True], dtype=bool)
        arr[mask] = np.array([r.a, r.b, r.c], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_assign_single_true_borrows_one(self):
        r = Region()
        r.x = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        mask = np.array([False, True, False], dtype=bool)
        arr[mask] = np.array([r.x], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 1)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 3 — local source, regional target (removing reference)
    def test_bool_assign_local_over_region_slots(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        arr[mask] = np.array([self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_assign_same_region_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        arr[mask] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_assign_cross_region_into_local_array_lrc_accounting(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True, False], dtype=bool)
        arr[mask] = np.array([r2.x], dtype=object)
        self.assertEqual(r1._lrc, base_lrc1 - 1)
        self.assertEqual(r2._lrc, base_lrc2 + 1)
        arr = None
        self.assertEqual(r1._lrc, base_lrc1 - 2)
        self.assertEqual(r2._lrc, base_lrc2)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_bool_assign_local_into_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        mask = np.array([True, False, True], dtype=bool)
        r.arr[mask] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        self.assertTrue(r.owns(local1))
        self.assertTrue(r.owns(local2))
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_assign_same_region_into_region_array_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False, True], dtype=bool)
        r.arr[mask] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_bool_assign_cross_region_into_region_array_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        mask = np.array([True, True], dtype=bool)
        with self.assertRaises(Exception):
            r1.arr[mask] = np.array([r2.x, r2.y], dtype=object)

    def test_bool_assign_cross_region_lrc_stable_after_failure(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True, True], dtype=bool)
        try:
            r1.arr[mask] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_bool_assign_partial_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        mask = np.array([False, True, False], dtype=bool)
        with self.assertRaises(Exception):
            r1.arr[mask] = np.array([r2.x], dtype=object)

    def test_bool_assign_partial_cross_region_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.c = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b, r1.c], dtype=object)
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([False, True, False], dtype=bool)
        try:
            r1.arr[mask] = np.array([r2.x], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # RHS: HAS_INTEGER
    def test_bool_lhs_integer_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        item = r.x
        self.assertEqual(r._lrc, base_lrc + 1)
        mask = np.array([True, False, False], dtype=bool)
        arr[mask] = np.array([item], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 1)
        item = None
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_lhs_integer_rhs_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        item0 = local_src[0]
        item1 = local_src[1]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = np.array([item0, item1], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 2)
        item0 = None
        item1 = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_lhs_integer_rhs_region_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        src = np.array([r.x, r.y], dtype=object)
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        item0 = src[0]
        item1 = src[1]
        self.assertEqual(r._lrc, base_lrc + 2)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = np.array([item0, item1], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 4)
        item0 = None
        item1 = None
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_lhs_integer_rhs_into_region_array_local(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        local = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, False], dtype=bool)
        r.arr[mask] = np.array([local], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 1)
        self.assertTrue(r.owns(local))
        local = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_lhs_integer_rhs_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        r2.x = self.A()
        r2.src = np.array([r2.x], dtype=object)
        item = r2.src[0]
        mask = np.array([True], dtype=bool)
        with self.assertRaises(Exception):
            r1.arr[mask] = np.array([item], dtype=object)

    def test_bool_lhs_integer_rhs_cross_region_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.arr = np.array([r1.a], dtype=object)
        r2.x = self.A()
        r2.src = np.array([r2.x], dtype=object)
        item = r2.src[0]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True], dtype=bool)
        try:
            r1.arr[mask] = np.array([item], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)
        item = None

    # RHS: HAS_SLICE
    def test_bool_lhs_slice_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        sliced = src[0:2]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = sliced
        self.assertEqual(r._lrc, base_lrc)
        sliced = None
        src = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 5)

    def test_bool_lhs_slice_rhs_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        sliced = local_src[0:2]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = sliced
        self.assertEqual(r._lrc, base_lrc - 2)
        sliced = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_lhs_slice_rhs_region_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        sliced = src[0:2]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = sliced
        self.assertEqual(r._lrc, base_lrc + 2)

    def test_bool_lhs_slice_rhs_into_region_array_local(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        local_src = np.array([local1, local2], dtype=object)
        sliced = local_src[0:2]
        mask = np.array([True, True], dtype=bool)
        r.arr[mask] = sliced
        self.assertEqual(r._lrc, base_lrc + 4)

    def test_bool_lhs_slice_rhs_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        r2.x = self.A()
        r2.y = self.A()
        r2.src = np.array([r2.x, r2.y], dtype=object)
        sliced = r2.src[0:2]
        mask = np.array([True, True], dtype=bool)
        with self.assertRaises(Exception):
            r1.arr[mask] = sliced

    def test_bool_lhs_slice_rhs_cross_region_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        r2.x = self.A()
        r2.y = self.A()
        r2.src = np.array([r2.x, r2.y], dtype=object)
        sliced = r2.src[0:2]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True, True], dtype=bool)
        try:
            r1.arr[mask] = sliced
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # RHS: HAS_ELLIPSIS
    def test_bool_lhs_ellipsis_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        ellipsis_view = src[...]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = ellipsis_view
        self.assertEqual(r._lrc, base_lrc)
        ellipsis_view = None
        src = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 5)

    def test_bool_lhs_ellipsis_rhs_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        ellipsis_view = local_src[...]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = ellipsis_view
        self.assertEqual(r._lrc, base_lrc - 2)
        ellipsis_view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_lhs_ellipsis_rhs_region_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        ellipsis_view = src[...]
        self.assertEqual(r._lrc, base_lrc)
        mask = np.array([True, True, False], dtype=bool)
        arr[mask] = ellipsis_view
        self.assertEqual(r._lrc, base_lrc + 2)

    def test_bool_lhs_ellipsis_rhs_into_region_array_local(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        local_src = np.array([local1, local2], dtype=object)
        ellipsis_view = local_src[...]
        mask = np.array([True, True], dtype=bool)
        r.arr[mask] = ellipsis_view
        self.assertEqual(r._lrc, base_lrc + 4)
        self.assertTrue(r.owns(local1))
        self.assertTrue(r.owns(local2))
        ellipsis_view = None
        local_src = None
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_lhs_ellipsis_rhs_cross_region_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        r2.x = self.A()
        r2.y = self.A()
        r2.src = np.array([r2.x, r2.y], dtype=object)
        ellipsis_view = r2.src[...]
        mask = np.array([True, True], dtype=bool)
        with self.assertRaises(Exception):
            r1.arr[mask] = ellipsis_view

    def test_bool_lhs_ellipsis_rhs_cross_region_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        r2.x = self.A()
        r2.y = self.A()
        r2.src = np.array([r2.x, r2.y], dtype=object)
        ellipsis_view = r2.src[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        mask = np.array([True, True], dtype=bool)
        try:
            r1.arr[mask] = ellipsis_view
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # RHS: HAS_BOOL
    def test_bool_lhs_bool_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        rhs_mask = np.array([True, True], dtype=bool)
        selected = src[rhs_mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        lhs_mask = np.array([True, True, False], dtype=bool)
        arr[lhs_mask] = selected
        self.assertEqual(r._lrc, base_lrc + 2)
        selected = None
        self.assertEqual(r._lrc, base_lrc)

    def test_bool_lhs_bool_rhs_local_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        rhs_mask = np.array([True, True], dtype=bool)
        selected = local_src[rhs_mask]
        lhs_mask = np.array([True, True, False], dtype=bool)
        arr[lhs_mask] = selected
        self.assertEqual(r._lrc, base_lrc - 2)
        selected = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_bool_lhs_bool_rhs_region_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        rhs_mask = np.array([True, True], dtype=bool)
        selected = src[rhs_mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        lhs_mask = np.array([True, True, False], dtype=bool)
        arr[lhs_mask] = selected
        self.assertEqual(r._lrc, base_lrc + 4)
        selected = None
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc)


# ===========================================================================
# array_assign_subscript — LHS: slice view (view = arr[i:j]; view[...] = val)
# ===========================================================================

class TestArrayAssign_SliceView(unittest.TestCase):
    """
    Tests for array_assign_subscript where the left-hand side is a slice view.
    Writes go through the view into the base buffer. The view itself holds no
    independent borrow on a local array; on a region-owned array the view is
    a local external reference (+1 LRC). All four RHS index types are covered.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 2 — local source, regional target (adding reference)
    def test_integer_assign_region_through_slice_view_over_local(self):
        r = Region()
        r.x = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[0:2]
        view[0] = r.x
        self.assertEqual(r._lrc, base_lrc + 1)
        arr = None
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_region_through_slice_view_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[0:2]
        view[0:2] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc + 2)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_same_region_through_slice_view_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc
        view = arr[0:2]
        view[0:2] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc)
        arr = None
        self.assertEqual(r._lrc, base_lrc)
        view = None
        self.assertEqual(r._lrc, base_lrc - 4)

    # Guideline 3 — local source, regional target (removing reference)
    def test_integer_assign_local_through_slice_view_decreases_lrc(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[0:2]
        local = self.A()
        view[0] = local
        self.assertEqual(r._lrc, base_lrc - 1)
        arr = None
        view = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_slice_assign_locals_through_slice_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc
        view = arr[1:3]
        view[0:2] = np.array([self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 2)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 2)
        view = None
        self.assertEqual(r._lrc, base_lrc - 4)

    def test_write_through_view_reflects_in_base(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        base_lrc = r._lrc
        arr = np.array([r.a, r.b, r.c], dtype=object)
        view = arr[0:2]
        local = self.A()
        view[0] = local
        view[1] = r.d
        lrc_after = r._lrc
        self.assertEqual(lrc_after, base_lrc + 3 - 1)
        view = None
        self.assertEqual(r._lrc, lrc_after)
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_integer_assign_local_through_slice_view_of_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[0:2]
        self.assertEqual(r._lrc, base_lrc + 1)
        local = self.A()
        view[0] = local
        self.assertEqual(r._lrc, base_lrc + 2)
        view = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local = None
        self.assertEqual(r._lrc, base_lrc)

    def test_integer_assign_region_through_slice_view_of_region_array_net_zero(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = r.arr[0:2]
        self.assertEqual(r._lrc, base_lrc + 1)
        view[0] = r.d
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_locals_through_slice_view_of_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local1 = self.A()
        local2 = self.A()
        view = r.arr[0:2]
        self.assertEqual(r._lrc, base_lrc + 1)
        view[0:2] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        self.assertEqual(r._lrc, base_lrc + 2)
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    def test_overlapping_views_writing_to_same_buffer(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.d = self.A()
        r.x = self.A()
        arr = np.array([r.a, r.b, r.c, r.d], dtype=object)
        base_lrc = r._lrc
        view1 = arr[0:3]
        view2 = arr[1:4]
        local = self.A()
        view1[1] = local
        self.assertEqual(r._lrc, base_lrc - 1)
        view2[0] = r.x
        self.assertEqual(r._lrc, base_lrc)
        view1 = None
        view2 = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 4)

    # Guideline 5 — cross-region raises
    def test_integer_assign_cross_region_through_slice_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[0:2]
        with self.assertRaises(Exception):
            view[0] = r2.x

    def test_slice_assign_cross_region_through_slice_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[0:2]
        with self.assertRaises(Exception):
            view[0:2] = np.array([r2.x, r2.y], dtype=object)

    def test_integer_assign_cross_region_through_slice_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[0:2]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0] = r2.x
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_slice_assign_cross_region_through_slice_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[0:2]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0:2] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # RHS: HAS_BOOL
    def test_slice_view_lhs_bool_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True], dtype=bool)
        selected = src[mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        view = arr[0:2]
        view[0:2] = selected
        self.assertEqual(r._lrc, base_lrc + 2)
        selected = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_view_lhs_bool_rhs_locals_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        mask = np.array([True, True], dtype=bool)
        selected = local_src[mask]
        view = arr[0:2]
        view[0:2] = selected
        self.assertEqual(r._lrc, base_lrc - 2)
        selected = None
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)


# ===========================================================================
# array_assign_subscript — LHS: ellipsis view (view = arr[...]; view[...] = val)
# ===========================================================================

class TestArrayAssign_EllipsisView(unittest.TestCase):
    """
    Tests for array_assign_subscript where the left-hand side is an ellipsis
    view variable. The view shares the base buffer identically to a full slice.
    When the base is a local array the view adds no LRC; when the base is
    region-owned the view is a local external reference (+1 LRC).
    All four RHS index types are covered.
    """

    def setUp(self):
        self.A = make_A()
        set_freezable(np.array([], dtype=np.float64).__class__, FREEZABLE_YES)
        freeze(np.array([], dtype=np.float64))

    # Guideline 2 — local source, regional target (adding reference)
    def test_integer_assign_region_through_ellipsis_view_over_local(self):
        r = Region()
        r.x = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0] = r.x
        self.assertEqual(r._lrc, base_lrc + 1)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_slice_assign_region_through_ellipsis_view_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0:2] = np.array([r.x, r.y], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 2)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_assign_region_through_ellipsis_view_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        r.z = self.A()
        arr = np.array([self.A(), self.A(), self.A()], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[...] = np.array([r.x, r.y, r.z], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 3 — local source, regional target (removing reference)
    def test_integer_assign_local_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        local = self.A()
        view[0] = local
        self.assertEqual(r._lrc, base_lrc - 1)
        view = None
        self.assertEqual(r._lrc, base_lrc - 1)
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_slice_assign_locals_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[0:2] = np.array([self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 2)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    def test_ellipsis_assign_locals_through_ellipsis_view(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.c = self.A()
        arr = np.array([r.a, r.b, r.c], dtype=object)
        base_lrc = r._lrc
        view = arr[...]
        view[...] = np.array([self.A(), self.A(), self.A()], dtype=object)
        self.assertEqual(r._lrc, base_lrc - 3)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 3)

    # Guideline 4 — regional source, local target (ephemeral move)
    def test_ellipsis_assign_locals_through_ellipsis_view_of_region_array(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        view = r.arr[...]
        self.assertEqual(r._lrc, base_lrc + 1)
        local1 = self.A()
        local2 = self.A()
        view[...] = np.array([local1, local2], dtype=object)
        self.assertEqual(r._lrc, base_lrc + 3)
        view = None
        self.assertEqual(r._lrc, base_lrc + 2)
        local1 = None
        self.assertEqual(r._lrc, base_lrc + 1)
        local2 = None
        self.assertEqual(r._lrc, base_lrc)

    # Guideline 5 — cross-region raises
    def test_integer_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[0] = r2.x

    def test_slice_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[0:2] = np.array([r2.x, r2.y], dtype=object)

    def test_ellipsis_assign_cross_region_through_ellipsis_view_raises(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        with self.assertRaises(Exception):
            view[...] = np.array([r2.x, r2.y], dtype=object)

    def test_integer_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0] = r2.x
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_slice_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[0:2] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    def test_ellipsis_assign_cross_region_through_ellipsis_view_lrc_stable(self):
        r1 = Region()
        r2 = Region()
        r1.a = self.A()
        r1.b = self.A()
        r2.x = self.A()
        r2.y = self.A()
        r1.arr = np.array([r1.a, r1.b], dtype=object)
        view = r1.arr[...]
        base_lrc1 = r1._lrc
        base_lrc2 = r2._lrc
        try:
            view[...] = np.array([r2.x, r2.y], dtype=object)
        except Exception:
            pass
        self.assertEqual(r1._lrc, base_lrc1)
        self.assertEqual(r2._lrc, base_lrc2)

    # RHS: HAS_BOOL
    def test_ellipsis_view_lhs_bool_rhs_same_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([r.a, r.b], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True], dtype=bool)
        selected = src[mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        view = arr[...]
        self.assertEqual(r._lrc, base_lrc + 2)
        view[...] = selected
        self.assertEqual(r._lrc, base_lrc + 2)
        selected = None
        self.assertEqual(r._lrc, base_lrc)

    def test_ellipsis_view_lhs_bool_rhs_locals_over_region(self):
        r = Region()
        r.a = self.A()
        r.b = self.A()
        arr = np.array([r.a, r.b], dtype=object)
        base_lrc = r._lrc
        local_src = np.array([self.A(), self.A()], dtype=object)
        mask = np.array([True, True], dtype=bool)
        selected = local_src[mask]
        view = arr[...]
        view[...] = selected
        self.assertEqual(r._lrc, base_lrc - 2)
        selected = None
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc - 2)

    def test_ellipsis_view_lhs_bool_rhs_region_over_locals(self):
        r = Region()
        r.x = self.A()
        r.y = self.A()
        arr = np.array([self.A(), self.A()], dtype=object)
        src = np.array([r.x, r.y], dtype=object)
        base_lrc = r._lrc
        mask = np.array([True, True], dtype=bool)
        selected = src[mask]
        self.assertEqual(r._lrc, base_lrc + 2)
        view = arr[...]
        view[...] = selected
        self.assertEqual(r._lrc, base_lrc + 4)
        selected = None
        self.assertEqual(r._lrc, base_lrc + 2)
        view = None
        arr = None
        self.assertEqual(r._lrc, base_lrc)


if __name__ == "__main__":
    unittest.main()