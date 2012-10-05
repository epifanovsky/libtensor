#ifndef LIBTENSOR_TOD_CONV_DIAG_TENSOR_H
#define LIBTENSOR_TOD_CONV_DIAG_TENSOR_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Converts diagonal tensors to dense tensors
    \tparam N Tensor order.

    This operation converts diagonal tensors to their dense representation by
    setting all irrelevant non-diagonal elements to zero.

    There are two perform() methods that do the conversion. One will map
    a diagonal tensor to a dense tensor, and the tensors must have the same
    dimensions. The other variant of perform() takes an offset as a second
    parameter and will project a diagonal tensor onto a window in a dense
    tensor. In that case the window must fully fit in the output dense tensor.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class tod_conv_diag_tensor : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    diag_tensor_rd_i<N, double> &m_ta; //!< Diagonal tensor

public:
    /** \brief Initializes the operation
     **/
    tod_conv_diag_tensor(diag_tensor_rd_i<N, double> &ta) : m_ta(ta) { }

    /** \brief Virtual destructor
     **/
    virtual ~tod_conv_diag_tensor() { }

    /** \brief Performs the conversion
        \param tb Output dense tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &tb);

    /** \brief Performs the conversion
        \param tb Output dense tensor.
        \param off Offset.
     **/
    void perform(dense_tensor_wr_i<N, double> &tb, const index<N> &off);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_TENSOR_H

