#include <pybind11/pybind11.h>

#include "Segment.h"

namespace py = pybind11;

float Segment::get_length() const
{
	return get_distance(a, b);
}

void init_segment_class(py::module& m)
{
    py::class_<Segment>(m, "Segment")
        .def(py::init<Point, Point>())
        .def_readwrite("a", &Segment::a)
        .def_readwrite("b", &Segment::b)
        .def("get_length", &Segment::get_length);
}
