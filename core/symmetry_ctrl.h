#ifndef LIBTENSOR_SYMMETRY_CTRL_H
#define LIBTENSOR_SYMMETRY_CTRL_H

#include "defs.h"
#include "exception.h"
#include "symmetry.h"

namespace libtensor {

template<size_t N, typename T>
class symmetry_ctrl {
public:
	typedef typename symmetry<N, T>::symmetry_element_t
		symmetry_element_t; //!< Symmetry element type

private:
	symmetry<N, T> &m_sym;

public:
	//!	\name Construction and destruction
	//@{

	symmetry_ctrl(symmetry<N, T> &sym);

	//@}


};


template<size_t N, typename T>
inline symmetry_ctrl<N, T>::symmetry_ctrl(symmetry<N, T> &sym) : m_sym(sym) {

}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_CTRL_H
