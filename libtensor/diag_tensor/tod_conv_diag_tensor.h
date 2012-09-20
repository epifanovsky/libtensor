#ifndef LIBTENSOR_TOD_CONV_DIAG_TENSOR_H
#define LIBTENSOR_TOD_CONV_DIAG_TENSOR_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Converts diagonal tensors to dense tensors
    \tparam N Tensor order.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class tod_conv_diag_tensor : public noncopyable {
private:
    diag_tensor_rd_i<N, double> &m_ta; //!< Diagonal tensor

public:
    /** \brief Initializes the operation
     **/
    tod_conv_diag_tensor(diag_tensor_rd_i<N, double> &ta) : m_ta(ta) { }

    /** \brief Performs the conversion
     **/
    void perform(dense_tensor_wr_i<N, double> &tb);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_TENSOR_H

