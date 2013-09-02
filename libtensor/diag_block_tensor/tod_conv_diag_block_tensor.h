#ifndef LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_H
#define LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include "diag_block_tensor_i.h"

namespace libtensor {


/** \brief Converts diagonal block tensors to dense tensors
    \tparam N Tensor order.

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N>
class tod_conv_diag_block_tensor : public noncopyable {
private:
    diag_block_tensor_rd_i<N, double> &m_bta;

public:
    tod_conv_diag_block_tensor(diag_block_tensor_rd_i<N, double> &bta) :
        m_bta(bta)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~tod_conv_diag_block_tensor() { }

    /** \brief Performs the conversion
        \param tb Output dense tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &tb);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_H
