#ifndef LIBTENSOR_TO_BTCONV_IMPL_H
#define LIBTENSOR_TO_BTCONV_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/block_tensor/bto_export.h>
#include "../dense_tensor_ctrl.h"
#include "../to_btconv.h"

namespace libtensor {


template<size_t N, typename T>
const char to_btconv<N, T>::k_clazz[] = "to_btconv<N, T>";


template<size_t N, typename T>
to_btconv<N, T>::to_btconv(block_tensor_rd_i<N, T> &bt) : m_bt(bt) {

}


template<size_t N, typename T>
to_btconv<N, T>::~to_btconv() {

}


template<size_t N, typename T>
void to_btconv<N, T>::perform(dense_tensor_wr_i<N, T> &t) {

    to_btconv<N, T>::start_timer();

    static const char method[] = "perform(dense_tensor_wr_i<N, T>&)";

    const block_index_space<N> &bis = m_bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());
    if(!bis.get_dims().equals(t.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "t");
    }

    dense_tensor_wr_ctrl<N, T> ctrl(t);
    T *ptr = ctrl.req_dataptr();
    bto_export<N, T>(m_bt).perform(ptr);
    ctrl.ret_dataptr(ptr);

    to_btconv<N, T>::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_TO_BTCONV_IMPL_H
