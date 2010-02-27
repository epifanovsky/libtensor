#ifndef LIBTENSOR_SO_PROJ_DOWN_H
#define LIBTENSOR_SO_PROJ_DOWN_H

#include "../core/mask.h"
#include "../core/symmetry_element_set.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_proj_down;

template<typename ElemT>
class so_proj_down_impl;

template<size_t N, size_t M, typename T>
struct symmetry_operation_params< so_proj_down<N, M, T> >;


/**	\brief Projection of a %symmetry group onto a subspace
	\tparam N Order of the argument space.
	\tparam M Decrement in the order of the result space.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space
	of order N - M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_proj_down;


/**	\brief Generic implementation of so_proj_down<N, T>

	This template provides the specification for the implementations of
	the so_proj_down<N, M, T> operation.

	\ingroup libtensor_symmetry
 **/
template<typename ElemT>
class so_proj_down_impl;


template<size_t N, size_t M, typename T>
struct symmetry_operation_params< so_proj_down<N, M, T> > {
public:
	const symmetry_element_set<N, T> &grp; //!< Symmetry group
	mask<N> msk; //!< Mask
	permutation<N - M> perm; //!< Permutation

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &grp_,
		const mask<N> &msk_,
		const permutation<N - M> &perm_) :

		grp(grp_), msk(msk_), perm(perm_) { }

	symmetry_operation_params(
		const symmetry_element_set<N, T> &grp_,
		const mask<N> &msk_) :

		grp(grp_), msk(msk_) { }
};


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_H