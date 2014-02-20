#ifndef LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
#define LIBTENSOR_SO_PERMUTE_IMPL_PERM_H

#include "../permutation_group.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    //  Adapter type for the input group
    //
    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> >
    adapter_t;

    adapter_t adapter1(params.g1);
    permutation_group<N, T> group(adapter1);
    group.permute(params.perm);
    params.g2.clear();
    group.convert(params.g2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
