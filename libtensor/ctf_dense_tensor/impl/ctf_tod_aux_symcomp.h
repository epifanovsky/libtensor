#ifndef LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H
#define LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H

#include <libtensor/ctf_dense_tensor/ctf_symmetry.h>

namespace libtensor {


template<size_t N, typename T>
size_t ctf_tod_aux_symcomp(const ctf_symmetry<N, T> &syma, size_t icompa,
    const ctf_symmetry<N, T> &symb) {

    int sa[N], sb[N];
    syma.write(icompa, sa);
    for(size_t icompb = 0; icompb != symb.get_ncomp(); icompb++) {
        symb.write(icompb, sb);
        bool same = true;
        for(size_t i = 0; i < N; i++) if(sa[i] != sb[i]) same = false;
        if(same) return icompb;
    }
    for(size_t icompb = 0; icompb != symb.get_ncomp(); icompb++) {
        symb.write(icompb, sb);
        bool good = true;
        for(size_t i = 0; i < N; i++) {
            if(!(sa[i] != NS && sb[i] == NS)) good = false;
        }
        if(good) return icompb;
    }
    return 0;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H

