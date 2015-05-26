#ifndef LIBTENSOR_CTF_BTOD_SET_IMPL_H
#define LIBTENSOR_CTF_BTOD_SET_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include "ctf_btod_set_symmetry.h"
#include "../ctf_btod_set.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_set<N>::k_clazz[] = "ctf_btod_set<N>";


template<size_t N>
void ctf_btod_set<N>::perform(ctf_block_tensor_i<N, double> &bta) {

    typedef ctf_block_tensor_i_traits<double> bti_traits;

    m_gbto.perform(bta);

    std::vector<size_t> blst;
    gen_block_tensor_ctrl<N, bti_traits> ctrl(bta);
    ctrl.req_nonzero_blocks(blst);
    ctf_btod_set_symmetry<N>().perform(blst, bta);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_IMPL_H
