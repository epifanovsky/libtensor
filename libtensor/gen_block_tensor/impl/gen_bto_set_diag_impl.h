#ifndef LIBTENSOR_GEN_BTO_SET_DIAG_IMPL_H
#define LIBTENSOR_GEN_BTO_SET_DIAG_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/orbit.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "../gen_bto_set_diag.h"
#include "../gen_block_tensor_ctrl.h"

namespace libtensor {


template<size_t N, typename Traits>
const char *gen_bto_set_diag<N, Traits>::k_clazz =
        "gen_bto_set_diag<N, Traits>";


template<size_t N, typename Traits>
gen_bto_set_diag<N, Traits>::gen_bto_set_diag(const element_type &v) : m_v(v) {

}


template<size_t N, typename Traits>
void gen_bto_set_diag<N, Traits>::perform(
        gen_block_tensor_i<N, bti_traits> &bt) {

    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_set_diag_type<N>::type to_set_diag;

    static const char *method =
            "perform(gen_block_tensor_i<N, bti_traits>&)";

    const block_index_space<N> &bis = bt.get_bis();
    size_t t = bis.get_type(0);
    for(size_t i = 1; i < N; i++) {
        if(bis.get_type(i) != t) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__,
                __LINE__, "Invalid block tensor dimension.");
        }
    }

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);

    dimensions<N> dims(bis.get_block_index_dims());
    size_t n = dims[0];
    index<N> idx;
    for(size_t i = 0; i < n; i++) {

        for(size_t j = 0; j < N; j++) idx[j] = i;

        abs_index<N> aidx(idx, dims);
        orbit<N, element_type> o(ctrl.req_const_symmetry(), idx);
        if(!o.is_allowed()) continue;
        if(o.get_acindex() != aidx.get_abs_index())
            continue;

        if(ctrl.req_is_zero_block(idx)) {
            if(! Traits::is_zero(m_v)) {
                wr_block_type &blk = ctrl.req_block(idx);
                to_set(Traits::zero()).perform(blk);
                to_set_diag(m_v).perform(blk);
                ctrl.ret_block(idx);
            }
        } else {
            wr_block_type &blk = ctrl.req_block(idx);
            to_set_diag(m_v).perform(blk);
            ctrl.ret_block(idx);
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_DIAG_H
