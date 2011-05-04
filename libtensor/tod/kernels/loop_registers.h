#ifndef LIBTENSOR_LOOP_REGISTERS_H
#define LIBTENSOR_LOOP_REGISTERS_H

#include <cstdlib> // for size_t

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
template<size_t N, size_t M>
struct loop_registers {
	const double *m_ptra[N]; //!< Position in argument arrays
	double *m_ptrb[M]; //!< Position in result arrays
	const double *m_ptra_end[N]; //!< End of argument arrays
	double *m_ptrb_end[M]; //!< End of result arrays
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_REGISTERS_H
