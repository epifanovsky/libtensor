#ifndef LIBTENSOR_SO_PROJ_UP_H
#define LIBTENSOR_SO_PROJ_UP_H

#include "../core/mask.h"
#include "../core/symmetry_element_set.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_proj_up;

template<typename ElemT>
class so_proj_up_impl;

template<size_t N, size_t M, typename T>
struct symmetry_operation_params< so_proj_up<N, M, T> >;


/**	\brief Projection of a %symmetry group to a larger space
	\tparam N Order of the argument space.
	\tparam M Increment in the order of the result space.

	The operation projects a given %symmetry group to a larger space.
	The resulting group will affect only the subspace of the large space
	specified by a %mask.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space of
	order N + M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_proj_up {
};


/**	\brief Generic implementation of so_proj_up<N, T>

	This template provides the specification for the implementations of
	the so_proj_up<N, M, T> operation.

	\ingroup libtensor_symmetry
 **/
template<typename ElemT>
class so_proj_up_impl {
};


template<size_t N, size_t M, typename T>
struct symmetry_operation_params< so_proj_up<N, M, T> > {
public:
	const symmetry_element_set<N, T> &grp; //!< Symmetry group
	mask<N + M> msk; //!< Mask
	permutation<N> perm; //!< Permutation

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &grp_,
		const mask<N + M> &msk_,
		const permutation<N> &perm_) :

		grp(grp_), msk(msk_), perm(perm_) { }

	symmetry_operation_params(
		const symmetry_element_set<N, T> &grp_,
		const mask<N + M> &msk_) :

		grp(grp_), msk(msk_) { }
};


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_H