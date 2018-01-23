#ifndef LIBTENSOR_LOOP_REGISTERS_H
#define LIBTENSOR_LOOP_REGISTERS_H

#include <cstdlib> // for size_t

namespace libtensor {


/** \brief Structure to keep track of the positions in input and output arrays
    \tparam N Number of input arrays.
    \tparam M Number of output arrays.

    \ingroup libtensor_kernels
 **/
template<size_t N, size_t M, typename T>
struct loop_registers_x {
    const T *m_ptra[N]; //!< Position in input arrays
    T *m_ptrb[M]; //!< Position in output arrays
    const T *m_ptra_end[N]; //!< End of input arrays (for overflow control)
    T *m_ptrb_end[M]; //!< End of output arrays (for overflow control)
};



template<size_t N, size_t M>
using loop_registers = loop_registers_x<N,M,double>;

} // namespace libtensor

#endif // LIBTENSOR_LOOP_REGISTERS_H
