#ifndef LIBTENSOR_TOD_SET_DIAG_H
#define LIBTENSOR_TOD_SET_DIAG_H

#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Assigns the diagonal elements of a tensor to a value
    \tparam N Tensor order.

    This operation sets the diagonal elements of a tensor to a value
    without affecting all the off-diagonal elements. The dimensions of the
    tensor must all be the same.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_set_diag {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Tensor element value (default 0.0).
     **/
    tod_set_diag(double v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
     **/
    void perform(dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_DIAG_H
