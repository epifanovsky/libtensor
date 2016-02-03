#ifndef LIBTENSOR_CTF_BTOD_RANDOM_IMPL_H
#define LIBTENSOR_CTF_BTOD_RANDOM_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_copy.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_random.h>
#include "ctf_btod_set_symmetry.h"
#include "../ctf_btod_random.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_random<N>::k_clazz[] = "ctf_btod_random<N>";


template<size_t N>
void ctf_btod_random<N>::perform(ctf_block_tensor_i<N, double> &bt) {

    typedef ctf_block_tensor_i_traits<double> bti_traits;

    m_gbto.perform(bt);

    std::vector<size_t> blst;
    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
    ctrl.req_nonzero_blocks(blst);
    ctf_btod_set_symmetry<N>().perform(blst, bt);
}


template<size_t N>
void ctf_btod_random<N>::perform(ctf_block_tensor_i<N, double> &bt,
    const index<N> &idx) {

    m_gbto.perform(bt, idx);

    std::vector<size_t> blst;
    dimensions<N> bidims = bt.get_bis().get_block_index_dims();
    blst.push_back(abs_index<N>::get_abs_index(idx, bidims));
    ctf_btod_set_symmetry<N>().perform(blst, bt);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_RANDOM_IMPL_H
