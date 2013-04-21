#ifndef LIBTENSOR_TOD_COPY_WND_H
#define LIBTENSOR_TOD_COPY_WND_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Copies a window from one tensor to another
    \tparam N Tensor order.

    \sa tod_copy

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_copy_wnd : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    dense_tensor_rd_i<N, double> &m_ta; //!< Source tensor
    index_range<N> m_ira; //!< Window 

public:
    /** \brief Prepares the operation
        \param ta Source tensor.
        \param ira Source window.
     **/
    tod_copy_wnd(dense_tensor_rd_i<N, double> &ta, const index_range<N> &ira);

    /** \brief Runs the operation
        \param tb Output tensor.
        \param trb Output window.
     **/
    void perform(dense_tensor_wr_i<N, double> &tb, const index_range<N> &irb);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_COPY_WND_H
