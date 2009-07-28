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
class default_symmetry : public symmetry_base< N, T, default_symmetry<N, T> > {
private:
	class bihandler : public block_iterator_handler_i<N, T> {
	public:
		virtual void on_begin(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit, const dimensions<N> &dims) const;
		virtual bool on_end(const index<N> &idx, const index<N> &orbit,
			const dimensions<N> &dims) const;
		virtual void on_next(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit, const dimensions<N> &dims) const;
	};

	class oihandler : public orbit_iterator_handler_i<N> {
	public:
		virtual void on_begin(index<N> &idx,
			const dimensions<N> &dims) const;
		virtual bool on_end(const index<N> &idx,
			const dimensions<N> &dims) const;
		virtual void on_next(index<N> &idx,
			const dimensions<N> &dims) const;
	};

private:
	oihandler m_oihandler; //!< Orbit iterator handler
	bihandler m_bihandler; //!< Block iterator handler

public:
	//!	\name Implementation of symmetry_i<N, T>
	//@{

	virtual void disable_symmetry();
	virtual void enable_symmetry();
	virtual orbit_iterator<N, T> get_orbits(const dimensions<N> &dims)
		const;

	//@}

};

template<size_t N, typename T>
void default_symmetry<N, T>::disable_symmetry() {

}

template<size_t N, typename T>
void default_symmetry<N, T>::enable_symmetry() {

}

template<size_t N, typename T>
orbit_iterator<N, T> default_symmetry<N, T>::get_orbits(
	const dimensions<N> &dims) const {

	return orbit_iterator<N, T>(m_oihandler, m_bihandler, dims);
}

template<size_t N, typename T>
void default_symmetry<N, T>::bihandler::on_begin(index<N> &idx,
	block_symop<N, T> &symop, const index<N> &orbit,
	const dimensions<N> &dims) const {

	idx = orbit;
}

template<size_t N, typename T>
bool default_symmetry<N, T>::bihandler::on_end(const index<N> &idx,
	const index<N> &orbit, const dimensions<N> &dims) const {

	return true;
}

template<size_t N, typename T>
void default_symmetry<N, T>::bihandler::on_next(index<N> &idx,
	block_symop<N, T> &symop, const index<N> &orbit,
	const dimensions<N> &dims) const {

}

template<size_t N, typename T>
void default_symmetry<N, T>::oihandler::on_begin(index<N> &idx,
	const dimensions<N> &dims) const {

}

template<size_t N, typename T>
bool default_symmetry<N, T>::oihandler::on_end(const index<N> &idx,
	const dimensions<N> &dims) const {

	for(register size_t i = 0; i < N; i++)
		if(idx[i] < dims[i] - 1) return false;
	return true;
}

template<size_t N, typename T>
void default_symmetry<N, T>::oihandler::on_next(index<N> &idx,
	const dimensions<N> &dims) const {

	dims.inc_index(idx);
}

} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_H
