#pragma once

#include "Point.h"

struct Segment 
{
	Segment() {}
	Segment(const Point& from, const Point& to, float dist = 0.0): a(from), b(to), length(dist) {}
	Segment(float a_x, float a_y, float b_x, float b_y, float dist = 0.0):a(a_x, a_y), b(b_x, b_y), length(dist) {}

	float get_length() const;

	Point a;
	Point b;
	float length;
};

