#ifndef LIBTENSOR_BTO_TRACE_IMPL_H
#define LIBTENSOR_BTO_TRACE_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>

namespace libtensor {


template<size_t N, typename Traits>
const char *bto_trace<N, Traits>::k_clazz = "bto_trace<N, Traits>";


template<size_t N, typename Traits>
bto_trace<N, Traits>::bto_trace(block_tensor_t &bta) : m_bta(bta) {
}


template<size_t N, typename Traits>
bto_trace<N, Traits>::bto_trace(block_tensor_t &bta,
    const permutation<k_ordera> &perm) : m_bta(bta), m_perm(perm) {

}


template<size_t N, typename Traits>
double bto_trace<N, Traits>::calculate() {

    typedef typename Traits::element_type element_t;

    typedef typename Traits::template block_type<k_ordera>::type block_t;

    typedef typename Traits::template block_tensor_ctrl_type<k_ordera>::type
        block_tensor_ctrl_t;

    typedef typename Traits::template to_trace_type<N>::type to_trace_t;

    static const char *method = "calculate()";

    element_t tr = 0;

    bto_trace<N, Traits>::start_timer();

    try {

    dimensions<k_ordera> bidimsa = m_bta.get_bis().get_block_index_dims();

    block_tensor_ctrl_t ca(m_bta);

    orbit_list<k_ordera, element_t> ola(ca.req_const_symmetry());
    for (typename orbit_list<k_ordera, element_t>::iterator ioa = ola.begin();
            ioa != ola.end(); ioa++) {

    if(ca.req_is_zero_block(ola.get_index(ioa))) continue;

    block_t *ba = 0;

    orbit<k_ordera, element_t> oa(ca.req_const_symmetry(), ola.get_index(ioa));
    for (typename orbit<k_ordera, element_t>::iterator iia = oa.begin();
            iia != oa.end(); iia++) {

        abs_index<k_ordera> aia(oa.get_abs_index(iia), bidimsa);
        index<k_ordera> ia(aia.get_index()); ia.permute(m_perm);

        bool skip = false;
        for(register size_t i = 0; i < N; i++) if(ia[i] != ia[N + i]) {
            skip = true;
            break;
        }
        if(skip) continue;

        tensor_transf<k_ordera, element_t> tra(oa.get_transf(iia));
        tra.permute(m_perm);

        if(ba == 0) ba = &ca.req_block(ola.get_index(ioa));
        element_t tr0 = to_trace_t(*ba, tra.get_perm()).calculate();
        tra.get_scalar_tr().apply(tr0);
        tr += tr0;
    }

    if(ba != 0) ca.ret_block(ola.get_index(ioa));

    }

    } catch(...) {
        bto_trace<N, Traits>::stop_timer();
        throw;
    }

    bto_trace<N, Traits>::stop_timer();

    return tr;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_TRACE_IMPL_H
