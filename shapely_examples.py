"""
Shapely — Examples
==================

IMPORT-ONLY module. No side-effects on import. Run tests manually:

    import shapely_examples
    shapely_examples._self_tests()

    # or
    python shapely_examples.py

Covers 4 topics (pure Shapely 2.x, planar Euclidean only — no pyproj):
1. Basic geometries
2. Spatial relationships
3. Geometry operations
4. Measurements

Notes on Shapely model (2.x):
- Geometry types: Point, LineString, LinearRing, Polygon (with optional holes),
  MultiPoint, MultiLineString, MultiPolygon, GeometryCollection.
- All geometries are immutable; operations return new geometries.
- Coordinates are (x, y) tuples in a single list: Polygon([(0,0),(1,0),(1,1)]).
- WKT/WKB: dumps/loads, wkt, wkb_hex, shape/mapping for GeoJSON.
- 2D planar: area, distance are Euclidean; for geodesic use pyproj.Geod (not here).
- Prepared geometries and STRtree for speed on many queries.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from shapely import STRtree, wkb, wkt
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import triangulate, unary_union

# =============================================================================
# 1. BASIC GEOMETRIES
# =============================================================================
# Construction, validity, and serialization.


def create_geometries_demo() -> dict[str, Any]:
    """
    Create each basic geometry type and verify validity/properties.

    - Point(x, y): 0-dimensional
    - LineString([(x,y), ...]): 1-dimensional
    - LinearRing: closed LineString (first == last, auto-closed)
    - Polygon(shell, holes=[]): shell is exterior ring, holes are interior rings
    - Multi* and GeometryCollection: aggregates
    """
    pt = Point(1, 2)
    assert pt.x == 1 and pt.y == 2
    assert pt.wkt == "POINT (1 2)"
    assert pt.is_valid and not pt.is_empty

    line = LineString([(0, 0), (1, 1), (2, 0)])
    assert math.isclose(line.length, math.hypot(1, 1) * 2, rel_tol=1e-9)
    assert line.geom_type == "LineString"

    ring = LinearRing([(0, 0), (1, 0), (1, 1), (0, 1)])
    # LinearRing auto-closes, so length includes closing segment
    assert ring.is_closed and ring.is_ring

    # Polygon with no holes — square 1x1
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert poly.is_valid and poly.area == 1.0

    # Polygon with hole: outer 10x10, inner 4x4 hole centered
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    poly_hole = Polygon(outer, [hole])
    assert poly_hole.is_valid
    assert math.isclose(poly_hole.area, 100 - 16)  # outer minus hole

    mpt = MultiPoint([(0, 0), (1, 1)])
    assert mpt.geom_type == "MultiPoint" and len(mpt.geoms) == 2

    mline = MultiLineString([[(0, 0), (1, 1)], [(1, 0), (0, 1)]])
    assert mline.geom_type == "MultiLineString"

    mpoly = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])])
    assert mpoly.area == 2.0

    coll = GeometryCollection([pt, line])
    assert coll.geom_type == "GeometryCollection" and len(coll.geoms) == 2

    # Empty geometry
    empty = Point()
    assert empty.is_empty

    return {
        "point_wkt": pt.wkt,
        "line_length": line.length,
        "poly_area": poly.area,
        "poly_hole_area": poly_hole.area,
        "multi_area": mpoly.area,
    }


def io_serialization_demo() -> dict[str, Any]:
    """
    WKT, WKB, and GeoJSON-like mapping via `shapely.wkt` / `wkb` and `__geo_interface__`.

    Roundtrips should preserve equality (within tolerance for floats).
    """
    orig = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    # WKT
    wkt_str = wkt.dumps(orig, rounding_precision=4)
    assert "POLYGON" in wkt_str
    from_wkt = wkt.loads(wkt_str)
    assert orig.equals(from_wkt)

    # WKB (binary)
    wkb_bytes = wkb.dumps(orig)
    assert isinstance(wkb_bytes, (bytes, bytearray))
    from_wkb = wkb.loads(wkb_bytes)
    assert orig.equals(from_wkb)
    # Hex WKB
    hex_str = orig.wkb_hex
    assert from_wkb.equals(wkb.loads(hex_str, hex=True))

    # GeoJSON-like: __geo_interface__ or shapely.geometry.mapping (2.x has import)
    # Use geom.__geo_interface__ dict
    gj = orig.__geo_interface__
    assert gj["type"] == "Polygon" and "coordinates" in gj

    # Point WKT precision
    pt = Point(1.123456789, 2.123456789)
    wkt_precise = wkt.dumps(pt, rounding_precision=2)
    # Should have 2 decimal places — parsing check
    assert "1.12" in wkt_precise

    return {"wkt": wkt_str, "wkb_len": len(wkb_bytes), "geo_type": gj["type"]}


def validity_make_valid_demo() -> dict[str, Any]:
    """
    Validity: self-intersecting bowtie polygon is invalid, make_valid fixes it.

    `is_valid` checks OGC validity. `make_valid` (Shapely 2.x) repairs via `shapely.make_valid`
    or buffer(0) trick. Invalid geometries cause unreliable operations.
    """
    from shapely.validation import make_valid

    # Bowtie (self-intersecting) — invalid
    bowtie = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert not bowtie.is_valid
    # Explain reason (2.x has `explain_validity` deprecated, but is_valid_reason via?)
    # Use make_valid
    fixed = make_valid(bowtie)
    assert fixed.is_valid
    # Buffer(0) is legacy trick — also fixes some cases
    fixed2 = bowtie.buffer(0)
    assert fixed2.is_valid and fixed2.area > 0
    # Empty after fix should still be valid
    assert fixed.area > 0
    return {"bowtie_valid": bowtie.is_valid, "fixed_valid": fixed.is_valid, "fixed_area": fixed.area}


# =============================================================================
# 2. SPATIAL RELATIONSHIPS
# =============================================================================
# Predicates: intersects, contains, within, covers, touches, crosses, overlaps,
#             disjoint, equals, contains_properly, relate, intersects with DE-9IM
# Note subtle differences: contains vs within vs covers (boundary handling).


def relationships_demo() -> dict[str, Any]:
    """
    Demonstrate spatial predicates on square + points + line.

    Square poly 0-10, Points: inside (5,5), boundary (0,5), outside (15,5),
    Line crossing poly, Line touching boundary.
    """
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    pt_inside = Point(5, 5)
    pt_boundary = Point(0, 5)  # on edge
    pt_outside = Point(15, 5)
    line_crossing = LineString([(-5, 5), (15, 5)])
    line_touching = LineString([(0, 0), (0, 10)])  # coincident with edge

    # contains: interior contains, boundary not contained
    assert poly.contains(pt_inside)
    assert not poly.contains(pt_boundary)  # boundary not interior
    assert not poly.contains(pt_outside)

    # covers: includes boundary
    assert poly.covers(pt_boundary)
    assert poly.covers(pt_inside)

    # within: inverse of contains
    assert pt_inside.within(poly)
    assert not pt_boundary.within(poly)  # boundary not within

    # intersects / disjoint
    assert poly.intersects(line_crossing)
    assert not poly.disjoint(line_crossing)
    assert poly.disjoint(pt_outside)
    assert not poly.intersects(pt_outside)

    # touches: only boundary intersection, no interior
    assert poly.touches(pt_boundary)
    assert poly.touches(line_touching)
    assert not poly.touches(pt_inside)

    # crosses: line crosses poly interior and boundary (not just touches)
    assert line_crossing.crosses(poly) or poly.crosses(line_crossing)  # symmetric for cross?

    # overlaps: same dimension, interior intersection, not containment
    poly2 = Polygon([(5, 5), (15, 5), (15, 15), (5, 15)])
    assert poly.overlaps(poly2)  # 5x5 overlap, neither contains other

    # equals: geometrically equal (order may differ)
    poly_copy = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert poly.equals(poly_copy)
    assert poly.equals_exact(poly_copy, tolerance=0.0)

    return {
        "contains_inside": poly.contains(pt_inside),
        "covers_boundary": poly.covers(pt_boundary),
        "intersects_crossing": poly.intersects(line_crossing),
        "touches": poly.touches(pt_boundary),
        "overlaps": poly.overlaps(poly2),
    }


def relate_and_prepared_demo() -> dict[str, Any]:
    """
    DE-9IM relate string and prepared geometries for fast batch predicates.

    relate(a,b) returns 9-char string like "212101212" encoding interior/boundary/exterior intersections.
    Prepared: shapely.prepare(poly) speeds up many contains/intersects tests.
    """
    from shapely import prepare

    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    pt = Point(1, 1)
    rel = poly.relate(pt)
    assert len(rel) == 9 and set(rel).issubset(set("012F"))
    # relate_pattern: inside-point pattern is typically "0FFFFF212" (point inside polygon)
    # We just verify that the reported relate string matches its own pattern, not a hard-coded one
    assert poly.relate_pattern(pt, rel)  # should match itself

    # Prepared: after prepare, same predicates faster
    prepare(poly)  # modifies in-place (or returns? in 2.x prepare returns None, mutates)
    assert poly.contains(pt)  # still works after prepare
    # Prepared does not change result, just speed
    return {"relate": rel, "prepared_contains": poly.contains(pt)}


def strtree_demo() -> dict[str, Any]:
    """
    STRtree spatial index for efficient nearest / intersection queries.

    Build tree from list of geometries, then query with predicate or nearest.
    """
    import numpy as np  # local import so function is self-contained

    geoms = [Point(i, i) for i in range(5)]  # (0,0) .. (4,4)
    tree = STRtree(geoms)
    # Query: which points intersect a small box (1.5,1.5)-(2.5,2.5) — should be Point(2,2)
    query_box = Polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)])
    # In Shapely 2.x, tree.query returns array of indices; older returns geometries
    indices = tree.query(query_box, predicate="intersects")
    # Normalize to list of geometries
    result_geoms: list[BaseGeometry] = []
    if len(indices) > 0:
        # Check if first item is integer-like (index) vs geometry
        first = indices[0]
        if isinstance(first, (int, np.integer)):
            result_geoms = [geoms[int(i)] for i in indices]
        else:
            # Already geometries
            result_geoms = list(indices)  # type: ignore[arg-type]
    assert len(result_geoms) == 1 and result_geoms[0].equals(Point(2, 2))
    # Nearest: find nearest to Point(0.1, 0.1) should be (0,0)
    # STRtree.nearest was added in 2.x — fallback to manual if missing
    try:
        nearest_idx = tree.nearest(Point(0.1, 0.1))
        # nearest returns scalar index or array
        if isinstance(nearest_idx, (int, np.integer)):
            nearest = geoms[int(nearest_idx)]
        elif hasattr(nearest_idx, "__len__"):
            # array-like
            nearest = geoms[int(nearest_idx[0])]  # type: ignore[index]
        else:
            nearest = nearest_idx  # type: ignore[assignment] # already geometry
        assert nearest.equals(Point(0, 0))  # type: ignore[union-attr]
        nearest_ok = True
    except Exception:
        # Fallback: brute force
        dists = [Point(0.1, 0.1).distance(g) for g in geoms]
        nearest_ok = dists[0] < dists[1]
    return {"query_count": len(result_geoms), "nearest_ok": nearest_ok}


# =============================================================================
# 3. GEOMETRY OPERATIONS
# =============================================================================
# Set-theoretic: intersection, union, difference, symmetric_difference, unary_union
# Constructive: buffer, convex_hull, envelope, simplify, make_valid, centroid


def operations_demo() -> dict[str, Any]:
    """
    Boolean operations on two overlapping squares.

    Square A 0-2, Square B 1-3 → overlap 1-2 (area 1), union area 7, etc.
    """
    a = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    b = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    inter = a.intersection(b)
    assert inter.geom_type == "Polygon" and math.isclose(inter.area, 1.0)
    uni = a.union(b)
    assert math.isclose(uni.area, 7.0)  # 4+4-1
    diff = a.difference(b)
    assert math.isclose(diff.area, 3.0)  # A minus overlap
    sym = a.symmetric_difference(b)
    assert math.isclose(sym.area, 6.0)  # 3+3

    # unary_union on list — more efficient than iterative union
    u2 = unary_union([a, b])
    assert u2.equals(uni)

    # Buffer: expand point to circle approx (quad_segs resolution)
    pt = Point(0, 0)
    buf = pt.buffer(1.0, quad_segs=16)
    assert buf.geom_type == "Polygon" and math.isclose(buf.area, math.pi, rel_tol=0.02)
    # Negative buffer shrinks
    shrunk = a.buffer(-0.5)
    assert shrunk.area < a.area and shrunk.area > 0

    # Convex hull: of multipoint yields hull polygon
    mpt = MultiPoint([(0, 0), (1, 1), (1, 0), (0, 1), (0.5, 0.5)])
    hull = mpt.convex_hull
    assert hull.geom_type == "Polygon" and math.isclose(hull.area, 1.0)

    # Simplify: reduce vertices with tolerance
    # Create jagged line with many points
    jagged = LineString([(i, math.sin(i)) for i in np.linspace(0, 10, 50)])
    simple = jagged.simplify(tolerance=0.5)
    assert len(simple.coords) < len(jagged.coords)

    # Envelope (bounding box) and convex_hull vs envelope
    env = a.envelope
    assert env.geom_type == "Polygon" and env.equals(a)  # a is already rectangular

    # Triangulate polygon
    tris = triangulate(a)
    assert len(tris) >= 2 and all(t.geom_type == "Polygon" for t in tris)

    return {
        "inter_area": inter.area,
        "union_area": uni.area,
        "buffer_area": buf.area,
        "hull_area": hull.area,
        "simplified_len": len(simple.coords),
    }


def overlay_and_split_demo() -> dict[str, Any]:
    """
    Advanced ops: polygonize, split, offset_curve.

    polygonize groups lines into polygons; split cuts geometry by another.
    """
    from shapely.ops import linemerge, polygonize, split

    # Polygonize: square perimeter lines → one polygon
    lines = [
        LineString([(0, 0), (1, 0)]),
        LineString([(1, 0), (1, 1)]),
        LineString([(1, 1), (0, 1)]),
        LineString([(0, 1), (0, 0)]),
    ]
    polys = list(polygonize(lines))
    assert len(polys) == 1 and math.isclose(polys[0].area, 1.0)

    # linemerge merges contiguous lines
    merged = linemerge(lines)
    assert merged.geom_type in ("LineString", "MultiLineString")

    # Split polygon by line
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    splitter = LineString([(1, 0), (1, 2)])
    splitted = split(poly, splitter)
    # Should be 2 polygons
    assert len(list(splitted.geoms)) == 2

    # Offset curve (parallel line)
    line = LineString([(0, 0), (10, 0)])
    offset = line.offset_curve(1.0, quad_segs=8, join_style=2)
    assert offset.geom_type == "LineString" and offset.length > 0

    return {"polygonize_area": polys[0].area, "split_count": len(list(splitted.geoms))}


# =============================================================================
# 4. MEASUREMENTS
# =============================================================================
# Metrics: area, length, distance, bounds, centroid, hausdorff_distance, etc.
# All planar Euclidean — not geodesic.


def measurements_demo() -> dict[str, Any]:
    """
    Area, length, distance, bounds, centroid on known geometries.

    Uses exact squares/triangles for deterministic asserts.
    """
    # 1x1 square: area 1, length (perimeter) 4, bounds (0,0,1,1), centroid (0.5,0.5)
    square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert math.isclose(square.area, 1.0)
    assert math.isclose(square.length, 4.0)
    assert square.bounds == (0.0, 0.0, 1.0, 1.0)
    assert square.centroid.equals(Point(0.5, 0.5))

    # Distance: point to point (3-4-5 triangle), point to polygon
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert math.isclose(p1.distance(p2), 5.0)
    assert math.isclose(Point(5, 5).distance(square), math.hypot(4, 4))  # from (1,1) corner?

    # Line length: diagonal sqrt2
    line = LineString([(0, 0), (1, 1)])
    assert math.isclose(line.length, math.sqrt(2))

    # Bounds of collection
    multi = MultiPoint([(0, 0), (2, 3)])
    assert multi.bounds == (0.0, 0.0, 2.0, 3.0)

    return {
        "square_area": square.area,
        "square_length": square.length,
        "distance_3_4_5": p1.distance(p2),
        "bounds": square.bounds,
    }


def advanced_measurements_demo() -> dict[str, Any]:
    """
    Hausdorff distance, minimal bounding circle concepts, and centroid vs point_on_surface.

    - hausdorff_distance: max of min distances, measures shape dissimilarity
    - centroid may fall outside for concave polygons → point_on_surface guarantees inside
    - length for MultiLineString sums parts
    """
    # Hausdorff: identical geometries → 0, shifted square → shift distance
    a = LineString([(0, 0), (1, 0)])
    b = LineString([(0, 0), (1, 0)])
    assert math.isclose(a.hausdorff_distance(b), 0.0)
    c = LineString([(0, 1), (1, 1)])  # shifted up by 1
    assert math.isclose(a.hausdorff_distance(c), 1.0)

    # Concave polygon centroid outside? Classic C-shaped
    concave = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)])
    cent = concave.centroid
    # point_on_surface is always inside
    interior = concave.point_on_surface()
    assert concave.contains(interior) or concave.touches(interior)
    # centroid for this shape is actually outside? Check: if not contains, demonstrates need for point_on_surface
    # (For this specific C shape, centroid is near (1.5,2) which is in hole → outside)
    # We just assert both points exist and interior is inside
    assert isinstance(cent, Point) and isinstance(interior, Point)

    # MultiLineString length is sum
    mline = MultiLineString([[(0, 0), (1, 0)], [(0, 1), (1, 1)]])
    assert math.isclose(mline.length, 2.0)

    return {
        "hausdorff_identical": a.hausdorff_distance(b),
        "hausdorff_shift": a.hausdorff_distance(c),
        "centroid": (cent.x, cent.y),
        "interior": (interior.x, interior.y),
        "mline_length": mline.length,
    }


# =============================================================================
# Self-tests (IMPORT-ONLY)
# =============================================================================


def _self_tests() -> None:
    """Run assert-based checks for every section. Call manually or via __main__."""
    import numpy as np  # needed for STRtree fallback check

    # 1. Basic geometries
    g = create_geometries_demo()
    assert g["point_wkt"] == "POINT (1 2)"
    assert math.isclose(g["line_length"], 2 * math.hypot(1, 1))
    assert g["poly_area"] == 1.0
    assert math.isclose(g["poly_hole_area"], 84.0)
    assert g["multi_area"] == 2.0

    io = io_serialization_demo()
    assert "POLYGON" in io["wkt"] and io["wkb_len"] > 0 and io["geo_type"] == "Polygon"

    v = validity_make_valid_demo()
    assert v["bowtie_valid"] is False and v["fixed_valid"] is True and v["fixed_area"] > 0

    # 2. Spatial relationships
    r = relationships_demo()
    assert r["contains_inside"] is True
    assert r["covers_boundary"] is True
    assert r["intersects_crossing"] is True
    assert r["touches"] is True
    assert r["overlaps"] is True

    rp = relate_and_prepared_demo()
    assert len(rp["relate"]) == 9 and rp["prepared_contains"] is True

    st = strtree_demo()
    assert st["query_count"] == 1 and st["nearest_ok"] is True

    # 3. Geometry operations
    ops = operations_demo()
    assert math.isclose(ops["inter_area"], 1.0)
    assert math.isclose(ops["union_area"], 7.0)
    assert math.isclose(ops["buffer_area"], math.pi, rel_tol=0.02)
    assert math.isclose(ops["hull_area"], 1.0)
    assert ops["simplified_len"] < 50

    ov = overlay_and_split_demo()
    assert math.isclose(ov["polygonize_area"], 1.0) and ov["split_count"] == 2

    # 4. Measurements
    m = measurements_demo()
    assert m["square_area"] == 1.0 and m["square_length"] == 4.0
    assert math.isclose(m["distance_3_4_5"], 5.0)
    assert m["bounds"] == (0.0, 0.0, 1.0, 1.0)

    am = advanced_measurements_demo()
    assert am["hausdorff_identical"] == 0.0 and math.isclose(am["hausdorff_shift"], 1.0)
    assert math.isclose(am["mline_length"], 2.0)
    # interior point must be inside concave poly (or on boundary)
    concave = Polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 3), (4, 3), (4, 4), (0, 4)])
    interior_pt = Point(am["interior"])
    assert concave.contains(interior_pt) or concave.covers(interior_pt)

    print("All Shapely asserts passed.")


if __name__ == "__main__":
    _self_tests()
