#include <pybind11/pybind11.h>

#include "Segment.h"

namespace py = pybind11;

/*
def on_same_side(seg_12, seg_34):
    """Checks pts 1 and 2 are on same side of seg_34 and vice-versa."""
    # If this function returns True, then the segments don't intersect.
    # If this function returns False, the segments might intersect (eg. collinear)

    def get_score(d_ba, b_ca):
        return d_ba.y * b_ca.x - d_ba.x * b_ca.y  # Eq of line

    p1, p2, p3, p4 = seg_12.a, seg_12.b, seg_34.a, seg_34.b
    # Check pts 3, 4 against seg_12
    delta_21, delta_31, delta_41 = p2 - p1, p3 - p1, p4 - p1
    score3, score4 = get_score(delta_21, delta_31), get_score(delta_21, delta_41)
    if score3 * score4 > 0:
        return True
    # Check pts 1, 2 against seg_34
    delta_43, delta_13, delta_23 = p4 - p3, p1 - p3, p2 - p3
    score1, score2 = get_score(delta_43, delta_13), get_score(delta_43, delta_23)
    if score1 * score2 > 0:
        return True
    return False

*/

int get_score(const Point& d_ba, const Point& b_ca)
{
	return d_ba.y * b_ca.x - d_ba.x * b_ca.y;  // Eq of line
}

bool on_same_side_native(const Segment& seg_12, const Segment& seg_34)
{
	const Point p1 = seg_12.a;
	const Point p2 = seg_12.b;
	const Point p3 = seg_34.a;
	const Point p4 = seg_34.b;

	// Check pts 3, 4 against seg_12
	const Point delta_21 = p2 - p1;
	const Point delta_31 = p3 - p1;
	const Point delta_41 = p4 - p1;

	int score3 = get_score(delta_21, delta_31);
	int score4 = get_score(delta_21, delta_41);

	if (score3 * score4 > 0)
	{
		return true;
	}
	else
	{
		Point delta_43 = p4 - p3;
		Point delta_13 = p1 - p3;
		Point delta_23 = p2 - p3;

		int score1 = get_score(delta_43, delta_13);
		int score2 = get_score(delta_43, delta_23);

		return (score1 * score2 > 0);
	}
}

bool on_same_side(const py::object& seg_12_obj, const py::object& seg_34_obj)
{
	Segment seg_12;
	Segment seg_34;

	seg_12.a.x = py::float_(seg_12_obj.attr("a").attr("x"));
	seg_12.a.y = py::float_(seg_12_obj.attr("a").attr("y"));
	seg_12.b.x = py::float_(seg_12_obj.attr("b").attr("x"));
	seg_12.b.y = py::float_(seg_12_obj.attr("b").attr("y"));

	seg_34.a.x = py::float_(seg_34_obj.attr("a").attr("x"));
	seg_34.a.y = py::float_(seg_34_obj.attr("a").attr("y"));
	seg_34.b.x = py::float_(seg_34_obj.attr("b").attr("x"));
	seg_34.b.y = py::float_(seg_34_obj.attr("b").attr("y"));

	return on_same_side_native(seg_12, seg_34);
}

void init_on_same_side(py::module_ &m) {
    m.def("on_same_side", &on_same_side, "Checks that the points in seg_12 are on the same side as that of seg_34 and vice-versa", py::arg("seg_12"), py::arg("seg_34"));
}
