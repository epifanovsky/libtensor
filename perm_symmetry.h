#ifndef LIBTENSOR_PERM_SYMMETRY_H
#define LIBTENSOR_PERM_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "symmetry_base.h"
#include "symmetry_operation_i.h"

namespace libtensor {

/**	\brief Permutational symmetry in block tensors


	\ingroup libtensor
 **/
template<size_t N, typename T>
class perm_symmetry : public symmetry_base< N, T, perm_symmetry<N, T> >,
	public symmetry_target< N, T, perm_symmetry<N, T> > {

public:
	virtual bool is_same_impl(const perm_symmetry<N, T> &other) const;
	virtual orbit_iterator<N, T> get_orbits(const dimensions<N> &dims);
	virtual void invoke(symmetry_operation_i<N, T> &op)
		throw(exception);

};

template<size_t N, typename T>
bool perm_symmetry<N, T>::is_same_impl(const perm_symmetry<N, T> &other) const {
	return true;
}

template<size_t N, typename T>
orbit_iterator<N, T> perm_symmetry<N, T>::get_orbits(
	const dimensions<N> &dims) {

	throw_exc("perm_symmetry<N, T>", "get_orbits()", "Not implemented");
}

template<size_t N, typename T>
void perm_symmetry<N, T>::invoke(symmetry_operation_i<N, T> &op)
	throw(exception) {
	op.perform(*this);
}

} // namespace libtensor

#endif // LIBTENSOR_PERM_SYMMETRY_H
