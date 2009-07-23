#ifndef LIBTENSOR_DEFAULT_SYMMETRY_H
#define LIBTENSOR_DEFAULT_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "symmetry_base.h"

namespace libtensor {

/**	\brief Default symmetry in block tensors (no symmetry)

	Simple implementation of empty symmetry. It provides no relationships
	among the blocks of a block %tensor making each block unique. Useful
	for testing and debugging.

	\ingroup libtensor
 **/
template<size_t N, typename T>
class default_symmetry : public symmetry_base< N, T, default_symmetry<N, T> >,
	public symmetry_target< N, T, default_symmetry<N, T> > {

public:
	//!	\name Implementation of
	//!		symmetry_target< N, T, default_symmetry<N, T> >
	//@{

	virtual bool is_same_impl(const default_symmetry<N, T> &other) const;

	//@}

	//!	\name Implementation of symmetry_i<N, T>
	//@{

	virtual void disable_symmetry();
	virtual void enable_symmetry();
	virtual orbit_iterator<N, T> get_orbits(
		const dimensions<N> &dims) const;

	//@}

};

template<size_t N, typename T>
bool default_symmetry<N, T>::is_same_impl(const default_symmetry<N, T> &other)
	const {

	return true;
}

template<size_t N, typename T>
void default_symmetry<N, T>::disable_symmetry() {

}

template<size_t N, typename T>
void default_symmetry<N, T>::enable_symmetry() {

}

template<size_t N, typename T>
orbit_iterator<N, T> default_symmetry<N, T>::get_orbits(
	const dimensions<N> &dims) const {

}

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_H
