#ifndef LIBTENSOR_SO_PERMUTE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_PERMUTE_SE_LABEL_IMPL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_permute.h"
#include "../se_label.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_permute<N, T>, se_label<N, T> >::k_clazz =
        "symmetry_operation_impl< so_permute<N, T>, se_label<N, T> >";

template<size_t N, typename T>
void
symmetry_operation_impl< so_permute<N, T>, se_label<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;
    adapter_t adapter1(params.grp1);

    params.grp2.clear();

    for (typename adapter_t::iterator it1 = adapter1.begin();
            it1 != adapter1.end(); it1++) {

        se_label<N, T> se2(adapter1.get_elem(it1));
        se2.permute(params.perm);

        params.grp2.insert(se2);
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_SE_LABEL_IMPL_H
