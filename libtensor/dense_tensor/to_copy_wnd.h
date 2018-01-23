#ifndef LIBTENSOR_TO_COPY_WND_H
#define LIBTENSOR_TO_COPY_WND_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Copies a window from one tensor to another
    \tparam N Tensor order.

    \sa to_copy

    \ingroup libtensor_dense_tensor_to
 **/
template<size_t N, typename T>
class to_copy_wnd : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    dense_tensor_rd_i<N, T> &m_ta; //!< Source tensor
    index_range<N> m_ira; //!< Window 

public:
    /** \brief Prepares the operation
        \param ta Source tensor.
        \param ira Source window.
     **/
    to_copy_wnd(dense_tensor_rd_i<N, T> &ta, const index_range<N> &ira);

    /** \brief Runs the operation
        \param tb Output tensor.
        \param trb Output window.
     **/
    void perform(dense_tensor_wr_i<N, T> &tb, const index_range<N> &irb);

};

template<size_t N>
using tod_copy_wnd = to_copy_wnd<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_TO_COPY_WND_H
