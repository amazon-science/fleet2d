#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_bbox_overlap(py::module_ &);
void init_on_same_side(py::module_ &);
void init_get_closest_point(py::module_ &);
void init_point_class(py::module_ &);
void init_segment_class(py::module_ &);

PYBIND11_MODULE(fov_utils_cpp, m)
{
	init_point_class(m);
	init_segment_class(m);

	init_bbox_overlap(m);
	init_on_same_side(m);
	init_get_closest_point(m);
}
