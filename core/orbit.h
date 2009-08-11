#ifndef LIBTENSOR_ORBIT_H
#define LIBTENSOR_ORBIT_H

#include "defs.h"
#include "exception.h"

namespace libtensor {

template<size_t N, typename T> class symmetry;


template<size_t N, typename T>
class orbit {
private:
	const symmetry<N, T> &m_sym; //!< Reference to parent symmetry
	size_t m_absidx; //!< Absolute %index of the canonical element

public:
	orbit(const symmetry<N, T> &sym, size_t absidx);
	size_t get_abs_index() const;
};


template<size_t N, typename T>
inline orbit<N, T>::orbit(const symmetry<N, T> &sym, size_t absidx)
: m_sym(sym), m_absidx(absidx) {

}


template<size_t N, typename T>
inline size_t orbit<N, T>::get_abs_index() const {
	return m_absidx;
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_H
