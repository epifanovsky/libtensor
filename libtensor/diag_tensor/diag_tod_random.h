#ifndef LIBTENSOR_DIAG_TOD_RANDOM_H
#define LIBTENSOR_DIAG_TOD_RANDOM_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Fills a tensor with random entries
    \param N Tensor order.

    This operation fills all the allowed entries in a diagonal tensor with
    random values in [0;1).

    \sa tod_random

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_random :
    public timings< diag_tod_random<N> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Initializes the operation
     **/
    diag_tod_random() { }

    /** \brief Performs the operation
        \param ta Output tensor.
     **/
    void perform(diag_tensor_wr_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_RANDOM_H

