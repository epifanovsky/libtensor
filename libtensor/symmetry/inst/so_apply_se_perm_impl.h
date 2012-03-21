#ifndef LIBTENSOR_SO_APPLY_SE_PERM_IMPL_H
#define LIBTENSOR_SO_APPLY_SE_PERM_IMPL_H

#include "../permutation_group.h"

namespace libtensor {

template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    //	Adapter type for the input group
    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;

    params.grp2.clear();

    permutation_group<N, T> group;

    adapter_t adapter(params.grp1);
    if (params.is_asym) {
        for (typename adapter_t::iterator it = adapter.begin();
                it != adapter.end(); it++) {

            const element_t &el = adapter.get_elem(it);
            if (el.is_symm())
                group.add_orbit(el.is_symm(), el.get_perm());
        }
    }
    else {
        for (typename adapter_t::iterator it = adapter.begin();
                it != adapter.end(); it++) {

            const element_t &el = adapter.get_elem(it);
            group.add_orbit(params.sign || el.is_symm(), el.get_perm());
        }
    }

    group.permute(params.perm1);
    group.convert(params.grp2);
}

} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_PERM_H
