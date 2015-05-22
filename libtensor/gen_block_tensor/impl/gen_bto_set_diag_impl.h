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
const char gen_bto_set_diag<N, Traits>::k_clazz[] =
        "gen_bto_set_diag<N, Traits>";


template<size_t N, typename Traits>
gen_bto_set_diag<N, Traits>::gen_bto_set_diag(
    const sequence<N, size_t> &msk, const element_type &v) :
    m_msk(msk), m_v(v) {

}


template<size_t N, typename Traits>
void gen_bto_set_diag<N, Traits>::perform(
    gen_block_tensor_i<N, bti_traits> &bt) {

    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_set_diag_type<N>::type to_set_diag;

    static const char method[] = "perform(gen_block_tensor_i<N, bti_traits>&)";

    const block_index_space<N> &bis = bt.get_bis();

    sequence<N, size_t> map(N);
    index<N> i1, i2;
    for (size_t i = 0; i < N; i++) {
        if (map[i] != N) continue;

        map[i] = i;
        size_t type = bis.get_type(i);
        i2[i] = bis.get_splits(type).get_num_points();

        if (m_msk[i] == 0) continue;

        for (size_t j = i + 1; j < N; j++) {
            if (m_msk[j] != m_msk[i]) continue;
            if (bis.get_type(i) != type) {
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Invalid block tensor dimension.");
            }
            map[j] = i;
        }
    }

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);

    dimensions<N> dims(index_range<N>(i1, i2));
    abs_index<N> ai(dims);
    do {
        const index<N> &idx = ai.get_index();
        for(size_t j = 0; j < N; j++) i1[j] = idx[map[j]];

        orbit<N, element_type> o(ctrl.req_const_symmetry(), i1);
        if(!o.is_allowed()) continue;
        if(o.get_cindex() != i1) continue;

        if(ctrl.req_is_zero_block(i1)) {
            if(! Traits::is_zero(m_v)) {
                wr_block_type &blk = ctrl.req_block(i1);
                to_set(Traits::zero()).perform(true, blk);
                to_set_diag(m_msk, m_v).perform(true, blk);
                ctrl.ret_block(i1);
            }
        } else {
            wr_block_type &blk = ctrl.req_block(i1);
            to_set_diag(m_msk, m_v).perform(true, blk);
            ctrl.ret_block(i1);
        }
    } while (ai.inc());
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SET_DIAG_H
