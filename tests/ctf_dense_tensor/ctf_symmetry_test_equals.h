#ifndef LIBTENSOR_CTF_SYMMETRY_TEST_EQUALS_H
#define LIBTENSOR_CTF_SYMMETRY_TEST_EQUALS_H

#include <libtensor/ctf_dense_tensor/ctf_symmetry.h>

namespace libtensor {


template<size_t N, typename T>
bool ctf_symmetry_test_equals(
    const ctf_symmetry<N, T> &sym1,
    const ctf_symmetry<N, T> &sym2) {

    const sequence<N, unsigned> &grp1 = sym1.get_grp();
    const sequence<N, unsigned> &indic1 = sym1.get_sym();
    const sequence<N, unsigned> &grp2 = sym2.get_grp();
    const sequence<N, unsigned> &indic2 = sym2.get_sym();

    unsigned g1 = grp1[0], g2 = grp2[0];
    for(size_t i = 0; i < N; i++) {
        if(indic1[g1] != indic2[g2]) return false;
        bool sameg1 = grp1[i] == g1, sameg2 = grp2[i] == g2;
        if(sameg1 != sameg2) return false;
        if(sameg1) continue;
        g1 = grp1[i];
        g2 = grp2[i];
    }
    return true;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_TEST_EQUALS_H
