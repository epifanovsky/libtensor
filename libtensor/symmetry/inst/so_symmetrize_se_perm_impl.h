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
    permutation_group<N, T> grp1(adapter1), grp2;

    size_t ngrp = 0, nidx = 0;
    bool has_zero = false;
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) { has_zero = true; continue; }
        ngrp = std::max(ngrp, params.idxgrp[i]);
        nidx = std::max(nidx, params.symidx[i]);
    }

    if (ngrp < 2) return;

    permutation<N> pp, cp;
    for (size_t k = 1; k < ngrp; k++) {
        for (size_t i = 1 ; i <= nidx; i++) {
            register size_t j = 0;
            for (; j < N; j++) {
                if ((params.idxgrp[j] == k) &&
                        (params.symidx[j] == i)) break;
            }
            size_t i1 = j;
            for (j = 0; j < N; j++) {
                if ((params.idxgrp[j] == k + 1) &&
                        (params.symidx[j] == i)) break;
            }
            if (k == 1) pp.permute(i1, j);
            cp.permute(i1, j);
        }
    }

    sequence<N, size_t> stabseq;
    size_t offset = (has_zero ? 1 : 0);
    for (register size_t i = 0; i < N; i++)
        stabseq[i] = params.idxgrp[i] + offset;

    grp1.stabilize(stabseq, grp2);
//    params.grp2.clear();
//    grp2.convert(params.grp2);

    if (ngrp > 2) grp2.add_orbit(params.symm || ((ngrp % 2) != 0), cp);
    grp2.add_orbit(params.symm, pp);

    params.grp2.clear();
    grp2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
