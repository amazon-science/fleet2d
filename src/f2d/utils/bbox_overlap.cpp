#include <pybind11/pybind11.h>
#include <iostream>
#include <algorithm>

#include "Segment.h"

namespace py = pybind11;

/*
    """Returns whether the bboxes of the two segments overlap."""
    min_pt_12x, max_pt_12x = min(seg_12.a.x, seg_12.b.x), max(seg_12.a.x, seg_12.b.x)
    min_pt_12y, max_pt_12y = min(seg_12.a.y, seg_12.b.y), max(seg_12.a.y, seg_12.b.y)

    min_pt_34x, max_pt_34x = min(seg_34.a.x, seg_34.b.x), max(seg_34.a.x, seg_34.b.x)
    min_pt_34y, max_pt_34y = min(seg_34.a.y, seg_34.b.y), max(seg_34.a.y, seg_34.b.y)

    if max_pt_34x < min_pt_12x or min_pt_34x > max_pt_12x:  # No overlap on x
        return False
    elif max_pt_34y < min_pt_12y or min_pt_34y > max_pt_12y:  # No overlap on y
        return False
    else:
        return True
*/
bool bbox_overlap_native(const Segment& seg_12, const Segment& seg_34)
{
	auto min_pt_12x = std::min(seg_12.a.x, seg_12.b.x);
	auto max_pt_12x = std::max(seg_12.a.x, seg_12.b.x);

	auto min_pt_12y = std::min(seg_12.a.y, seg_12.b.y);
	auto max_pt_12y = std::max(seg_12.a.y, seg_12.b.y);

	auto min_pt_34x = std::min(seg_34.a.x, seg_34.b.x);
	auto max_pt_34x = std::max(seg_34.a.x, seg_34.b.x);

	auto min_pt_34y = std::min(seg_34.a.y, seg_34.b.y);
	auto max_pt_34y = std::max(seg_34.a.y, seg_34.b.y);

	return !(
		(max_pt_34x < min_pt_12x || min_pt_34x > max_pt_12x) ||
		(max_pt_34y < min_pt_12y || min_pt_34y > max_pt_12y)
	);
}

bool bbox_overlap(const py::object& seg_12_obj, const py::object& seg_34_obj)
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

	return bbox_overlap_native(seg_12, seg_34);
}

void init_bbox_overlap(py::module_ &m) {
    m.def("bbox_overlap", &bbox_overlap, "Determine if the bounding box of two given segments overlap ", py::arg("seg_12"), py::arg("seg_34"));
}
