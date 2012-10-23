#ifndef LIBTENSOR_DIAG_BTOD_COPY_IMPL_H
#define LIBTENSOR_DIAG_BTOD_COPY_IMPL_H

#include <libtensor/diag_tensor/diag_tod_copy.h>
#include <libtensor/diag_tensor/diag_tod_set.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_add.h>
#include <libtensor/gen_block_tensor/gen_bto_aux_copy.h>
#include "../diag_btod_copy.h"

namespace libtensor {


template<size_t N>
const char *diag_btod_copy<N>::k_clazz = "diag_btod_copy<N>";


template<size_t N>
void diag_btod_copy<N>::perform(diag_block_tensor_i<N, double> &btb) {

    gen_bto_aux_copy<N, diag_btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N>
void diag_btod_copy<N>::perform(
    diag_block_tensor_i<N, double> &btb,
    const double &c) {

    typedef diag_block_tensor_i_traits<double> bti_traits;

    gen_block_tensor_ctrl<N, bti_traits> cb(btb);
    addition_schedule<N, diag_btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    gen_bto_aux_add<N, diag_btod_traits> out(get_symmetry(), asch, btb,
        scalar_transf<double>(c));
    perform(out);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_COPY_IMPL_H
