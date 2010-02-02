#ifndef LIBTENSOR_SO_INTERSECTION_H
#define LIBTENSOR_SO_INTERSECTION_H

#include "../core/symmetry_element_set.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_intersection;

template<typename ElemT>
class so_intersection_impl;

template<size_t N, typename T>
struct symmetry_operation_params< so_intersection<N, T> >;


/**	\brief Intersection of two %symmetry groups

	The intersection of two %symmetry groups \f$ G = G_A \cap G_B \f$
	combines the elements present in both groups such that the result
	is also a group.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_intersection {
};


/**	\brief Generic implementation of so_intersection<N, T>

	This template provides the specification for the implementations of
	the so_intersection<N, T> operation.

	\ingroup libtensor_symmetry
 **/
template<typename ElemT>
class so_intersection_impl {
};


template<size_t N, typename T>
struct symmetry_operation_params< so_intersection<N, T> > {
public:
	const symmetry_element_set<N, T> &g1;
	const symmetry_element_set<N, T> &g2;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<N, T> &g2_) : g1(g1_), g2(g2_) { }
};


} // namespace libtensor

#endif // LIBTENSOR_SO_INTERSECTION_H

