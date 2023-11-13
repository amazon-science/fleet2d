#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "Segment.h"
#include "ThreadPool.h"

namespace py = pybind11;

#define NUM_WORKERS	2
std::unique_ptr<ThreadPool> GLOBAL_POOL;

bool bbox_overlap(const py::object& seg_12, const py::object& seg_34);
bool on_same_side(const py::object& seg_12, const py::object& seg_34);
bool bbox_overlap_native(const Segment& seg_12, const Segment& seg_34);
bool on_same_side_native(const Segment& seg_12, const Segment& seg_34);

void _get_seg_vars(const Segment& seg, double vars[3])
{
	vars[0] = (seg.a.y - seg.b.y);
	vars[1] = (seg.b.x - seg.a.x);
	vars[2] = (seg.b.x * seg.a.y - seg.a.x * seg.b.y);
}

/**
 * Returns intersection of two segments (nullptr if they don't intersect).
 */
std::shared_ptr<Point> get_intersection(const Point& source_pt, const Segment& source_seg, const Segment& other_seg)
{
	// We get to this function once we know the segments do not lie on the same
	// side of each other. Thus, they are collinear or they intersect.
	double source_v[3];
	_get_seg_vars(source_seg, source_v);

	double other_v[3];
	_get_seg_vars(other_seg, other_v);

	double determinant = source_v[0] * other_v[1] - source_v[1] * other_v[0];
	if (determinant == 0)
	{
		// They are collinear (possibly non-overlapping)
		double dist_other_a = get_distance(source_pt, other_seg.a);
		double dist_other_b = get_distance(source_pt, other_seg.b);

		const Point* other_pt;
		if ( dist_other_a < dist_other_b)
		{
			other_pt = &other_seg.a;
		}
		else
		{
			other_pt = &other_seg.b;
		}
		double other_dist = std::min(dist_other_a, dist_other_b);
		if (other_dist > get_distance(source_pt, source_seg.b))
		{
			//  No intersection (non-overlapping collinear segments)
			return nullptr;
		}
		else
		{
			return std::make_shared<Point>(*other_pt);
		}
	}

	double dx = source_v[2] * other_v[1] - source_v[1] * other_v[2];
	double dy = source_v[0] * other_v[2] - source_v[2] * other_v[0];

	return std::make_shared<Point>(dx / determinant, dy / determinant);
}

bool _check_segment(std::size_t i, const py::object& ray_from_src_obj, const std::vector<py::object>& segment_objs, const py::dict& segment_to_attrs,  double& max_dist, Point& closest_point)
{
	const py::object& curr_seg_obj = segment_objs.data()[i];
	double np_distance = py::float_(segment_to_attrs[curr_seg_obj]["np_distance"]);

	if (np_distance >= max_dist)
	{
		// std::cout << i << ". " << np_distance << " >= " << max_dist << ", skipping" << std::endl;
		return false;
	}
	else if ( !bbox_overlap(ray_from_src_obj, curr_seg_obj) )
	{
		// std::cout << i << ". " << "No bounding box overlap for " << ray_from_src_obj << " and " << curr_seg_obj << std::endl;
		return false;
	}
	else if ( on_same_side(ray_from_src_obj, curr_seg_obj) )
	{
		// std::cout << i << ". " << "ray_from_src and curr_seg are on the same side" << std::endl;
		return false;
	}
	else
	{
		Segment curr_seg;
		curr_seg.a.x = py::float_(curr_seg_obj.attr("a").attr("x"));
		curr_seg.a.y = py::float_(curr_seg_obj.attr("a").attr("y"));
		curr_seg.b.x = py::float_(curr_seg_obj.attr("b").attr("x"));
		curr_seg.b.y = py::float_(curr_seg_obj.attr("b").attr("y"));

		Segment ray_from_src;
		ray_from_src.a.x = py::float_(ray_from_src_obj.attr("a").attr("x"));
		ray_from_src.a.y = py::float_(ray_from_src_obj.attr("a").attr("y"));
		ray_from_src.b.x = py::float_(ray_from_src_obj.attr("b").attr("x"));
		ray_from_src.b.y = py::float_(ray_from_src_obj.attr("b").attr("y"));

		const Point& source_pt = ray_from_src.a;

		std::shared_ptr<Point> int_pt = get_intersection(source_pt, ray_from_src, curr_seg);

		if (!int_pt)
		{
			// std::cout << i << ". " << "No intersection point between ray_from_src and curr_seg when viewed from source_pt" << std::endl;
			return false;
		}
		else
		{
			double dist_from_source = get_distance(*int_pt, source_pt);

			// std::cout << i << ". " << "Found intersection point " << int_pt->x << ", " << int_pt->y << std::endl;
			// std::cout << "distance from source " << source_pt.x << ", " << source_pt.y << " = " << dist_from_source << std::endl;
			if (dist_from_source >= max_dist )
			{
				// std::cout << i << ". " << "dist_from_source " << dist_from_source << " is greater than max_dist " << max_dist << std::endl;
				return false;
			}
			else
			{
				// std::cout << i << ". " << "Closest point is now @ " << i << " w/ distance " << dist_from_source << " which is less than " << max_dist << std::endl;
				closest_point = *int_pt;
				max_dist = dist_from_source;

				return true;
			}
		}
	}
}

bool _check_segment_native(std::size_t i, const Segment& ray_from_src, const std::vector<Segment>& segments, double& max_dist, Point& closest_point)
{
	const Segment& curr_seg = segments.data()[i];

	if (curr_seg.length > max_dist)
	{
		// std::cout << i << ". " << "Length " << curr_seg.length << " is > " << max_dist << std::endl;
		return false;
	}
	else if ( !bbox_overlap_native(ray_from_src, curr_seg) )
	{
		return false;
	}
	else if ( on_same_side_native(ray_from_src, curr_seg) )
	{
		return false;
	}
	else
	{
		const Point& source_pt = ray_from_src.a;

		std::shared_ptr<Point> int_pt = get_intersection(source_pt, ray_from_src, curr_seg);
		if (!int_pt)
		{
			// std::cout << i << ". " << "No intersection point between ray_from_src and curr_seg when viewed from source_pt" << std::endl;
			return false;
		}
		else
		{
			double dist_from_source = get_distance(*int_pt, source_pt);

			// std::cout << i << ". " << "Found intersection point " << int_pt->x << ", " << int_pt->y << std::endl;
			// std::cout << "distance from source " << source_pt.x << ", " << source_pt.y << " = " << dist_from_source << std::endl;
			if (dist_from_source >= max_dist )
			{
				// std::cout << i << ". " << "dist_from_source " << dist_from_source << " is greater than max_dist " << max_dist << std::endl;
				return false;
			}
			else
			{
				// std::cout << i << ". " << "Closest point is now @ " << i << " w/ distance " << dist_from_source << " which is less than " << max_dist << std::endl;
				closest_point = *int_pt;
				max_dist = dist_from_source;

				return true;
			}
		}
	}
}

std::tuple<py::object, py::object> get_closest_point(const py::object& source_pt_obj, const py::object& ray_from_src_obj, const std::vector<py::object>& segment_objs, double max_dist, const py::dict& segment_to_attrs, std::size_t prev_seg_idx)
{
	Point closest_point;
	std::size_t best_seg_idx = prev_seg_idx;

	// std::cout << "### Starting search for " << source_pt_obj << ", " << ray_from_src_obj << ", vector of size " << segments.size() << std::endl;
	bool found = _check_segment(prev_seg_idx, ray_from_src_obj, segment_objs, segment_to_attrs, max_dist, closest_point);

	for ( std::size_t i = 0; i < prev_seg_idx; ++i )
	{
		if ( _check_segment(i, ray_from_src_obj, segment_objs, segment_to_attrs, max_dist, closest_point) )
		{
			best_seg_idx = i;
			found = true;
		}
	}

	for ( std::size_t i = prev_seg_idx + 1; i < segment_objs.size(); ++i )
	{
		if ( _check_segment(i, ray_from_src_obj, segment_objs, segment_to_attrs, max_dist, closest_point) )
		{
			best_seg_idx = i;
			found = true;
		}
	}

	// std::cout << "### Ending search for " << source_pt_obj << ", " << ray_from_src_obj << ", vector of size " << segments.size() << std::endl;
	// std::cout << "### Best segment for " << source_pt_obj << ", " << ray_from_src_obj << " is " << segments[best_seg_idx] << " w/ closest point = " << closest_point.x << ", " << closest_point.y << " @ distance " << max_dist << std::endl;
	if ( !found )
	{
		return std::make_tuple(py::cast<py::none>(Py_None), py::cast<py::none>(Py_None));
	}
	else
	{
		return std::make_tuple<py::object, py::object>(py::cast(closest_point), py::cast(best_seg_idx));
	}
}

struct CheckSegmentResult
{
	CheckSegmentResult(double max_dist, const Point& closest_point, std::size_t best_seg_idx, bool found):
		max_dist(max_dist), closest_point(closest_point), best_seg_idx(best_seg_idx), found(found) {}

	double max_dist;
	Point closest_point;
	std::size_t best_seg_idx;
	bool found;
};

std::unique_ptr<CheckSegmentResult> _check_segments_thread(const Segment& ray_from_src,
				const std::vector<Segment>& segments,
				double max_dist, Point closest_point, std::size_t best_seg_idx, bool found,
				std::size_t start_idx, std::size_t end_idx)
{
	// std::cout << "Check segments thread starting from " << start_idx << " to " << end_idx << std::endl;
	found = false;
	for ( std::size_t i = start_idx; i < end_idx; ++i )
	{
		if ( _check_segment_native(i, ray_from_src, segments, max_dist, closest_point) )
		{
			best_seg_idx = i;
			found = true;
		}
	}
	// std::cout << "Check segments thread done from " << start_idx << " to " << end_idx << std::endl;
	return std::make_unique<CheckSegmentResult>(max_dist, closest_point, best_seg_idx, found);
}

std::tuple<py::object, py::object> get_closest_point_mt(const py::object& source_pt_obj, const py::object& ray_from_src_obj, const std::vector<py::object>& segment_objs, double max_dist, const py::dict& segment_to_attrs, std::size_t prev_seg_idx)
{
	Point closest_point;
	std::size_t best_seg_idx = prev_seg_idx;

	Segment ray_from_src;
	ray_from_src.a.x = py::float_(ray_from_src_obj.attr("a").attr("x"));
	ray_from_src.a.y = py::float_(ray_from_src_obj.attr("a").attr("y"));
	ray_from_src.b.x = py::float_(ray_from_src_obj.attr("b").attr("x"));
	ray_from_src.b.y = py::float_(ray_from_src_obj.attr("b").attr("y"));

	// XXX: I suspect this to be a large bottleneck, there are typically around 700 segments
	// to be converted from Python to Segment objects in order to leverage parallelism
	//
	// Will need to figure out a way to have Python use the Segment C++ class directly
	//
	std::size_t segments_size = segment_objs.size();
	std::vector<Segment> segments;
	segments.reserve(segments_size);
	for (std::size_t i = 0; i < segments_size; ++i)
	{
		const py::object& curr_seg_obj = segment_objs.data()[i];
		float a_x = py::float_(curr_seg_obj.attr("a").attr("x"));
		float a_y = py::float_(curr_seg_obj.attr("a").attr("y"));
		float b_x = py::float_(curr_seg_obj.attr("b").attr("x"));
		float b_y = py::float_(curr_seg_obj.attr("b").attr("y"));

		double np_distance = py::float_(segment_to_attrs[curr_seg_obj]["np_distance"]);

		segments.emplace_back(a_x, a_y, b_x, b_y, np_distance);
	}

	// std::cout << "### Starting search for " << ray_from_src_obj << ", vector of size " << segments.size() << std::endl;
	bool found = _check_segment_native(prev_seg_idx, ray_from_src, segments, max_dist, closest_point);

	std::vector<std::future<std::unique_ptr<CheckSegmentResult>>> results(NUM_WORKERS);
	std::size_t start_idx = 0, end_idx = 0, idx_increment = segments_size / NUM_WORKERS;
	for ( std::size_t i = 0; i < NUM_WORKERS - 1; ++i )
	{
		end_idx += idx_increment;

		// std::cout << "Enqueuing " << i << ". " << start_idx << " to " << end_idx << std::endl;
		results[i] = GLOBAL_POOL->enqueue(_check_segments_thread,
					std::ref(ray_from_src),
					std::ref(segments),
					max_dist, closest_point, best_seg_idx, found,
					start_idx, end_idx);

		start_idx = end_idx;

	}
	// std::cout << "Enqueuing " << (NUM_WORKERS - 1) << ". " << end_idx << " to " << segments_size << std::endl;
	results[NUM_WORKERS - 1] = GLOBAL_POOL->enqueue(_check_segments_thread,
					std::ref(ray_from_src),
					std::ref(segments),
					max_dist, closest_point, best_seg_idx, found,
					end_idx, segments_size);

	// Wait for all threads to finish
	//
	for (std::size_t i = 0; i < NUM_WORKERS; ++i) { results[i].wait(); }


	// std::cout << "All workers finished" << std::endl;
	CheckSegmentResult best_result(max_dist, closest_point, prev_seg_idx, found);
	for (std::size_t i = 0; i < NUM_WORKERS; ++i)
	{
		auto curr_result = results[i].get();
		// std::cout << "For batch " << i << " the result " << curr_result.get() << " is @ index " << curr_result->best_seg_idx << " w/ distance = " << curr_result->max_dist << " w/ found = " << curr_result->found << " vs prev_seg_idx " << prev_seg_idx << std::endl;
		if (curr_result->found && curr_result->max_dist < best_result.max_dist )
		{
			// std::cout << "Best result is found in batch " << i << std::endl;
			best_result.max_dist = curr_result->max_dist;
			best_result.closest_point = curr_result->closest_point;
			best_result.best_seg_idx = curr_result->best_seg_idx;
			best_result.found = true;
		}
	}

	max_dist = best_result.max_dist;
	closest_point = best_result.closest_point;
	best_seg_idx = best_result.best_seg_idx;
	found = best_result.found;

	// std::cout << "### Ending search for " << source_pt_obj << ", " << ray_from_src_obj << ", vector of size " << segments.size() << " w/ found = " << found << std::endl;
	// std::cout << "### Best segment for " << source_pt_obj << ", " << ray_from_src_obj << " is " << best_seg_idx << " = " << segment_objs[best_seg_idx] << " w/ closest point = " << closest_point.x << ", " << closest_point.y << " @ distance " << max_dist << std::endl;
	if ( !found )
	{
		// std::cout << "#### NOT FOUND ####" << std::endl;
		return std::make_tuple(py::cast<py::none>(Py_None), py::cast<py::none>(Py_None));
	}
	else
	{
		return std::make_tuple<py::object, py::object>(py::cast(closest_point), py::cast(best_seg_idx));
	}
}

void init_get_closest_point(py::module_ &m) {
	m.def("get_closest_point", &get_closest_point,
		"Filters a given set of segments to exclude segments that do not overlap in the given ray and are not on the same side as the ray",
		py::arg("source_pt"),
		py::arg("ray_from_src"),
		py::arg("segments"),
		py::arg("max_dist"),
		py::arg("segment_to_attrs"),
		py::arg("prev_seg_idx"));

	m.def("get_closest_point_mt", &get_closest_point_mt,
		"Filters a given set of segments to exclude segments that do not overlap in the given ray and are not on the same side as the ray",
		py::arg("source_pt"),
		py::arg("ray_from_src"),
		py::arg("segments"),
		py::arg("max_dist"),
		py::arg("segment_to_attrs"),
		py::arg("prev_seg_idx"));

	GLOBAL_POOL = std::make_unique<ThreadPool>(NUM_WORKERS);
}
