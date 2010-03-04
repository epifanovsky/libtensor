#ifndef LIBTENSOR_SO_UNION_H
#define LIBTENSOR_SO_UNION_H

#include "../core/symmetry_element_set.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_union;

template<typename ElemT>
class so_union_impl;

template<size_t N, typename T>
class symmetry_operation_params< so_union<N, T> >;


/**	\brief Union of two %symmetry groups

	The union of two %symmetry groups \f$ G = G_A \cup G_B \f$ combines
	the elements of the groups such that the result is also a group.
	If the union is not a group, the operation causes an exception.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_union {
};


/**	\brief Generic implementation of so_union<N, T>

	This template provides the specification for the implementations of
	the so_union<N, T> operation.

	\ingroup libtensor_symmetry
 **/
template<typename ElemT>
class so_union_impl {
};


template<size_t N, typename T>
class symmetry_operation_params< so_union<N, T> > {
public:
	const symmetry_element_set<N, T> &g1;
	const symmetry_element_set<N, T> &g2;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<N, T> &g2_) : g1(g1_), g2(g2_) { }
};


} // namespace libtensor

#endif // LIBTENSOR_SO_UNION_H

