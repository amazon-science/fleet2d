#pragma once

struct Point
{
	Point(): x(0.0f), y(0.0f) {}
	Point(double x_coord, double y_coord): x(x_coord), y(y_coord) {}
	Point(const Point& right): x(right.x), y(right.y) {}

	double x;
	double y;
};

double get_distance(const Point& a, const Point& b);

Point operator*(const Point& left, float scale);
Point operator+(const Point& left, const Point& right);
Point operator-(const Point& left, const Point& right);
bool operator<(const Point& left, const Point& right);
