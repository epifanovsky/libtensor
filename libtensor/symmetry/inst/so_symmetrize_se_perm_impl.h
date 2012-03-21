#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H

#include "../permutation_group.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;

    adapter_t adapter1(params.grp1);
    permutation_group<N, T> grp2(adapter1);
    grp2.add_orbit(params.symm, params.perm);

    params.grp2.clear();
    grp2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
