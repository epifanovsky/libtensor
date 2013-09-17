#ifndef LIBTENSOR_SO_APPLY_SE_PART_IMPL_H
#define LIBTENSOR_SO_APPLY_SE_PART_IMPL_H

#include <libtensor/core/abs_index.h>

namespace libtensor {

template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;

    params.grp2.clear();

    adapter_t adapter1(params.grp1);

    for (typename adapter_t::iterator it1 = adapter1.begin();
            it1 != adapter1.end(); it1++) {

        const element_t &se1 = adapter1.get_elem(it1);
        const dimensions<N> &pdims = se1.get_pdims();
        element_t se2(se1.get_bis(), pdims);

        abs_index<N> ai(pdims);
        do {

            const index<N> &i1 = ai.get_index();
            if (se1.is_forbidden(i1) && params.keep_zero) {
                se2.mark_forbidden(i1); continue;
            }

            index<N> i2 = se1.get_direct_map(i1);
            while (i1 < i2) {

                scalar_transf<T> tr = se1.get_transf(i1, i2);
                if (tr.is_identity()) {
                    se2.add_map(i1, i2, tr);
                    break;
                }
                else if (tr == params.s1) {
                    se2.add_map(i1, i2, params.s2);
                    break;
                }

                i2 = se1.get_direct_map(i2);
            }
        } while (ai.inc());

        se2.permute(params.perm1);
        params.grp2.insert(se2);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_SE_PART_IMPL_H
