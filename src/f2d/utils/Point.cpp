#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include <cmath>

#include "Point.h"

namespace py = pybind11;

double get_distance(const Point& a, const Point& b)
{
	return std::sqrt(std::pow(a.x - b.x , 2) + std::pow(a.y - b.y, 2));
}

Point operator+(const Point& left, const Point& right)
{
	Point sum(left.x + right.x, left.y + right.y);

	return sum;
}

Point operator-(const Point& left, const Point& right)
{
	Point diff(left.x - right.x, left.y - right.y);

	return diff;
}

Point operator*(const Point& left, double scale)
{
	Point mult(left.x * scale, left.y * scale);

	return mult;
}

bool operator<(const Point& left, const Point& right)
{
	return (left.x == right.x) ?
		(left.y < right.y) :
		(left.x < left.y);
}

void init_point_class(py::module& m)
{
	py::class_<Point>(m, "Point")
		.def(py::init<double, double>())
		.def(py::init<Point>())
		.def(py::self + py::self)
		.def(py::self - py::self)
		.def(py::self * double())
		.def(py::self < py::self)
		.def_readwrite("x", &Point::x)
		.def_readwrite("y", &Point::y)
		.def("__iter__", [](const Point &p) { return py::make_iterator(&p.x, &p.y); }, py::keep_alive<0, 1>());
}
