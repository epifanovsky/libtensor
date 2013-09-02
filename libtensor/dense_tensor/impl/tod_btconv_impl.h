#ifndef LIBTENSOR_TOD_BTCONV_IMPL_H
#define LIBTENSOR_TOD_BTCONV_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/block_tensor/btod_export.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_btconv.h"

namespace libtensor {


template<size_t N>
const char tod_btconv<N>::k_clazz[] = "tod_btconv<N>";


template<size_t N>
tod_btconv<N>::tod_btconv(block_tensor_rd_i<N, double> &bt) : m_bt(bt) {

}


template<size_t N>
tod_btconv<N>::~tod_btconv() {

}


template<size_t N>
void tod_btconv<N>::perform(dense_tensor_wr_i<N, double> &t) {

    tod_btconv<N>::start_timer();

    static const char method[] = "perform(dense_tensor_wr_i<N, double>&)";

    const block_index_space<N> &bis = m_bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());
    if(!bis.get_dims().equals(t.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *ptr = ctrl.req_dataptr();
    btod_export<N>(m_bt).perform(ptr);
    ctrl.ret_dataptr(ptr);

    tod_btconv<N>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_IMPL_H
