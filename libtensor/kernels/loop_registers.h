#ifndef LIBTENSOR_LOOP_REGISTERS_H
#define LIBTENSOR_LOOP_REGISTERS_H

#include <cstdlib> // for size_t

namespace libtensor {


/** \brief Structure to keep track of the positions in input and output arrays
    \tparam N Number of input arrays.
    \tparam M Number of output arrays.

    \ingroup libtensor_kernels
 **/
template<size_t N, size_t M>
struct loop_registers {
    const double *m_ptra[N]; //!< Position in input arrays
    double *m_ptrb[M]; //!< Position in output arrays
    const double *m_ptra_end[N]; //!< End of input arrays (for overflow control)
    double *m_ptrb_end[M]; //!< End of output arrays (for overflow control)
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_REGISTERS_H
