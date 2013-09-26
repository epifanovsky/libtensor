#ifndef LIBTENSOR_CTF_BTOD_COLLECT_IMPL_H
#define LIBTENSOR_CTF_BTOD_COLLECT_IMPL_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_collect.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "../ctf_btod_collect.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_collect<N>::k_clazz[] = "ctf_btod_collect<N>";


template<size_t N>
void ctf_btod_collect<N>::perform(block_tensor_wr_i<N, double> &bt) {

    static const char method[] = "perform(block_tensor_wr_i<N, double>&)";

    if(!m_dbt.get_bis().equals(bt.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt");
    }

    typedef ctf_block_tensor_i_traits<double> ctf_bti_traits;
    typedef block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_rd_ctrl<N, ctf_bti_traits> ca(m_dbt);
    gen_block_tensor_wr_ctrl<N, bti_traits> cb(bt);

    cb.req_zero_all_blocks();
    so_copy<N, double>(ca.req_const_symmetry()).perform(cb.req_symmetry());

    dimensions<N> bidims(m_dbt.get_bis().get_block_index_dims());

    std::vector<size_t> nzblk;
    ca.req_nonzero_blocks(nzblk);

    for(size_t i = 0; i < nzblk.size(); i++) {
        index<N> idx;
        abs_index<N>::get_index(nzblk[i], bidims, idx);
        ctf_dense_tensor_i<N, double> &blka = ca.req_const_block(idx);
        dense_tensor_wr_i<N, double> &blkb = cb.req_block(idx);
        ctf_tod_collect<N>(blka).perform(blkb);
        cb.ret_block(idx);
        ca.ret_const_block(idx);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_COLLECT_IMPL_H

