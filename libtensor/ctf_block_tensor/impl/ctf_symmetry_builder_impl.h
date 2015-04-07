#ifndef LIBTENSOR_CTF_SYMMETRY_BUILDER_IMPL_H
#define LIBTENSOR_CTF_SYMMETRY_BUILDER_IMPL_H

#include <algorithm>
#include "../ctf_symmetry_builder.h"

namespace libtensor {


template<size_t N, typename T>
ctf_symmetry<N, T> ctf_symmetry_builder<N, T>::build(
    const transf_list<N, T> &trl) {

    sequence<N, unsigned> grp(0), grpind(0);
    for(size_t i = 0; i < N; i++) grp[i] = i;

    for(typename transf_list<N, T>::iterator i = trl.begin();
        i != trl.end(); ++i) {

        const tensor_transf<N, T> &tr = trl.get_transf(i);

        sequence<N, size_t> seq1, seq2;
        for(size_t j = 0; j < N; j++) seq1[j] = seq2[j] = j;
        tr.get_perm().apply(seq2);
        size_t ndiff = 0, jdiff;
        for(size_t j = 0; j < N; j++) if(seq1[j] != seq2[j]) {
            jdiff = j;
            ndiff++;
        }
        if(ndiff != 2) continue; // Skip all non-pairwise permutations

        sequence<N, unsigned> grp2(grp);
        tr.get_perm().apply(grp2);
        unsigned g1 = grp[jdiff], g2 = grp2[jdiff];
        unsigned symasym = 0;
        if(tr.get_scalar_tr().get_coeff() == T(-1)) symasym = 1;
        if(g1 == g2) continue; // Already the same group of indices
        if(g1 > g2) std::swap(g1, g2);
        for(size_t j = 0; j < N; j++) if(grp[j] == g2) grp[j] = g1;
        grpind[g1] = symasym;
    }

    return ctf_symmetry<N, T>(grp, grpind);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_BUILDER_IMPL_H
