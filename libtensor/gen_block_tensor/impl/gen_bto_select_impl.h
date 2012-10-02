#ifndef LIBTENSOR_GEN_BTO_SELECT_IMPL_H
#define LIBTENSOR_GEN_BTO_SELECT_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_copy.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_select.h"


namespace libtensor {


template<size_t N, typename Traits, typename ComparePolicy>
const char *gen_bto_select<N, Traits, ComparePolicy>::k_clazz =
        "gen_bto_select<N, Traits, ComparePolicy>";


template<size_t N, typename Traits, typename ComparePolicy>
gen_bto_select<N, Traits, ComparePolicy>::gen_bto_select(
        gen_block_tensor_rd_i<N, bti_traits> &bt, compare_type cmp) :
    m_bt(bt), m_sym(m_bt.get_bis()), m_cmp(cmp) {

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);
    so_copy<N, element_type>(ctrl.req_const_symmetry()).perform(m_sym);

}

template<size_t N, typename Traits, typename ComparePolicy>
gen_bto_select<N, Traits, ComparePolicy>::gen_bto_select(
        gen_block_tensor_rd_i<N, bti_traits> &bt,
        const symmetry<N, element_type> &sym, compare_type cmp) :
    m_bt(bt), m_sym(m_bt.get_bis()), m_cmp(cmp) {

    static const char *method =
            "gen_bto_select(gen_block_tensor_rd_i<N, bti_traits>, "
            "const symmetry<N, element_type> &, compare_type)";

    if (! m_sym.get_bis().equals(sym.get_bis()))
        throw bad_parameter(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Invalid symmetry.");

    so_copy<N, element_type>(sym).perform(m_sym);
}


template<size_t N, typename Traits, typename ComparePolicy>
void gen_bto_select<N, Traits, ComparePolicy>::perform(
        list_type &li, size_t n) {

    static const char *method = "perform(list_type &, size_t)";

    if (n == 0) return;
    li.clear();

    const block_index_space<N> &bis = m_bt.get_bis();
    dimensions<N> bidims(bis.get_block_index_dims());

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);
    const symmetry<N, element_type> &sym = ctrl.req_const_symmetry();

    // Loop over all orbits of imposed symmetry
    orbit_list<N, element_type> ol(m_sym);
    for (typename orbit_list<N, element_type>::iterator iol = ol.begin();
            iol != ol.end(); iol++) {

        index<N> idxa, idxa0;
        abs_index<N>::get_index(ol.get_abs_index(iol), bidims, idxa);

        orbit<N, element_type> oa(sym, idxa);
        if (! oa.is_allowed()) continue;

        abs_index<N>::get_index(oa.get_abs_canonical_index(), bidims, idxa0);
        if (ctrl.req_is_zero_block(idxa0)) continue;

        // Obtain block
        rd_block_type &t = ctrl.req_const_block(idxa0);

        const tensor_transf<N, element_type> &tra = oa.get_transf(idxa);

        // Create element list for canonical block (within the symmetry)
        to_list_type tlc;
        to_select(t, tra, m_cmp).perform(tlc, n);
        merge_lists(li, idxa, tlc, n);

        ctrl.ret_const_block(idxa0);
    }
}

template<size_t N, typename Traits, typename ComparePolicy>
void gen_bto_select<N, Traits, ComparePolicy>::merge_lists(list_type &to,
        const index<N> &bidx, const to_list_type &from, size_t n) {

    typename list_type::iterator ibt = to.begin();
    for (typename to_list_type::const_iterator it = from.begin();
            it != from.end(); it++) {

        while (ibt != to.end()) {
            if (m_cmp(it->get_value(), ibt->get_value())) break;
            ibt++;
        }

        if (to.size() == n && ibt == to.end()) {
            return;
        }

        ibt = to.insert(ibt, block_tensor_element_type(bidx, *it));
        if (to.size() > n) to.pop_back();
        ibt++;
    }
}

} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SELECT_IMPL_H
