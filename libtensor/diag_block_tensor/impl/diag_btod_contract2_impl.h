#ifndef LIBTENSOR_DIAG_BTOD_CONTRACT_IMPL_H
#define LIBTENSOR_DIAG_BTOD_CONTRACT_IMPL_H

#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../diag_btod_contract2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char *diag_btod_contract2<N, M, K>::k_clazz =
    "diag_btod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
void diag_btod_contract2<N, M, K>::perform(
    diag_block_tensor_i<NC, double> &btc) {

    gen_bto_aux_copy<NC, diag_btod_traits> out(get_symmetry(), btc);
    perform(out);
}


template<size_t N, size_t M, size_t K>
void diag_btod_contract2<N, M, K>::perform(
    diag_block_tensor_i<NC, double> &btc,
    const double &d) {

    typedef diag_block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_ctrl<NC, bti_traits> cc(btc);
    addition_schedule<NC, diag_btod_traits> asch(get_symmetry(),
        cc.req_const_symmetry());
    asch.build(get_schedule(), cc);

    gen_bto_aux_add<NC, diag_btod_traits> out(get_symmetry(), asch, btc,
        scalar_transf<double>(d));
    perform(out);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_CONTRACT2_IMPL_H
