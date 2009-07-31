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
	private:
		dimensions<N> m_dims;
	public:
		bihandler(const dimensions<N> &dims) : m_dims(dims) { };
		virtual ~bihandler() { };
		virtual bool on_begin(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit) const;
		virtual bool on_next(index<N> &idx, block_symop<N, T> &symop,
			const index<N> &orbit) const;
	};

	class oihandler : public orbit_iterator_handler_i<N, T> {
	private:
		dimensions<N> m_dims;
	public:
		oihandler(const dimensions<N> &dims) : m_dims(dims) { };
		virtual ~oihandler() { };
		virtual bool on_begin(index<N> &idx) const;
		virtual bool on_next(index<N> &idx) const;
	};

private:
	oihandler m_oihandler; //!< Orbit iterator handler
	bihandler m_bihandler; //!< Block iterator handler

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates symmetry within given dimensions
		\param dims Block %index dimensions
	 **/
	default_symmetry(const dimensions<N> &dims);

	/**	\brief Virtual destructor
	 **/
	virtual ~default_symmetry() { };

	//@}

	//!	\name Implementation of symmetry_i<N, T>
	//@{

	virtual void disable_symmetry();
	virtual void enable_symmetry();
	virtual bool is_canonical(const index<N> &idx) const;
	virtual const orbit_iterator_handler_i<N, T> &get_oi_handler() const;
	virtual const block_iterator_handler_i<N, T> &get_bi_handler() const;

	//@}

};


template<size_t N, typename T>
default_symmetry<N, T>::default_symmetry(const dimensions<N> &dims)
: m_oihandler(dims), m_bihandler(dims) {

}


template<size_t N, typename T>
void default_symmetry<N, T>::disable_symmetry() {

}


template<size_t N, typename T>
void default_symmetry<N, T>::enable_symmetry() {

}


template<size_t N, typename T>
bool default_symmetry<N, T>::is_canonical(const index<N> &idx) const {

	return true;
}


template<size_t N, typename T>
const orbit_iterator_handler_i<N, T> &default_symmetry<N, T>::get_oi_handler()
	const {

	return m_oihandler;
}


template<size_t N, typename T>
const block_iterator_handler_i<N, T> &default_symmetry<N, T>::get_bi_handler()
	const {

	return m_bihandler;
}


template<size_t N, typename T>
bool default_symmetry<N, T>::bihandler::on_begin(index<N> &idx,
	block_symop<N, T> &symop, const index<N> &orbit) const {

	idx = orbit;
	return true;
}


template<size_t N, typename T>
bool default_symmetry<N, T>::bihandler::on_next(index<N> &idx,
	block_symop<N, T> &symop, const index<N> &orbit) const {

	return false;
}


template<size_t N, typename T>
bool default_symmetry<N, T>::oihandler::on_begin(index<N> &idx) const {

	return true;
}


template<size_t N, typename T>
bool default_symmetry<N, T>::oihandler::on_next(index<N> &idx) const {

	return m_dims.inc_index(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYMMETRY_H
