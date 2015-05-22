#ifndef LIBTENSOR_TOD_SET_DIAG_H
#define LIBTENSOR_TOD_SET_DIAG_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/sequence.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Alters the elements of a generalized diagonal of a tensor by a
        constant value
    \tparam N Tensor order.

    This operation sets or shifts the elements of a generalized diagonal of a
    tensor by a constant value without affecting the off-diagonal elements.
    The generalized diagonal is defined using a masking sequence. Dimensions
    for which the mask is 0 are not part of the diagonal, while dimensions
    for which the mask has the same non-zero value are taken as diagonal.
    For example, using the operation on a tensor \f$ a\f$ with mask [10221]
    will only affect the elements \f$ a_{ijkki}, \forall i, j, k \f$ .

    The dimensions of the tensor belonging to the same diagonal must be
    identical.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_set_diag : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    sequence<N, size_t> m_msk; //!< Diagonal mask
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param msk Diagonal mask
        \param v Tensor element value (default 0.0).
     **/
    tod_set_diag(const sequence<N, size_t> &msk, double v = 0.0) :
        m_msk(msk), m_v(v) { }

    /** \brief Initializes the operation (compatability wrapper)
        \param v Tensor element value (default 0.0).
     **/
    tod_set_diag(double v = 0.0) : m_msk(1), m_v(v) { }

    /** \brief Performs the operation
        \param zero Zero out diagonal first
        \param t Result tensor
     **/
    void perform(bool zero, dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_DIAG_H
