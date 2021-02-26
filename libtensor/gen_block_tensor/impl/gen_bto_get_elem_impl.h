#ifndef LIBTENSOR_GEN_BTO_GET_ELEM_IMPL_H
#define LIBTENSOR_GEN_BTO_GET_ELEM_IMPL_H

#include <libtensor/exception.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_get_elem.h"
#include <libtensor/block_tensor/block_tensor_i_traits.h>
#include <libtensor/dense_tensor/to_get_elem.h>

namespace libtensor {


template<size_t N, typename Traits>
const char *gen_bto_get_elem<N, Traits>::k_clazz =
        "gen_bto_get_elem<N, Traits>";


template<size_t N, typename Traits>
void gen_bto_get_elem<N, Traits>::perform(
        gen_block_tensor_i<N, bti_traits> &bt, const index<N> &bidx,
        const index<N> &idx, element_type &d) {

    //typedef typename Traits::template to_get_type<N>::type to_get;
    //typedef typename Traits::template to_get_elem_type<N>::type to_get_elem<>;

    static const char *method = "perform("
            "gen_block_tensor_rd_i<N, bti_traits> &, const index<N> &, "
            "const index<N> &, const element_type &)";

    gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    orbit<N, element_type> o(ctrl.req_const_symmetry(), bidx);

    if (! o.is_allowed())
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Block index not allowed by symmetry.");

    const tensor_transf<N, element_type> &tr = o.get_transf(bidx);
    abs_index<N> abidx(o.get_acindex(), bidims);

    bool zero = ctrl.req_is_zero_block(abidx.get_index());

    if(zero) {
        d = (element_type)0.0; 
        return;
    }

    rd_block_type &blk = ctrl.req_const_block(abidx.get_index());

    permutation<N> perm(tr.get_perm(), true); // TODO: validate correctness!!!
    index<N> idx1(idx); idx1.permute(perm);
    to_get_elem<N, element_type>().perform(blk, idx1, d);
    tr.get_scalar_tr().apply(d);


    ctrl.ret_const_block(abidx.get_index());
}




} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_GET_ELEM_IMPL_H
