#ifndef LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H
#define LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H

#include <utility>
#include <libtensor/ctf_dense_tensor/ctf_symmetry.h>

namespace libtensor {


template<size_t N, typename T>
std::pair<bool, size_t> ctf_tod_aux_symcomp_ex(const ctf_symmetry<N, T> &syma,
    size_t icompa, const ctf_symmetry<N, T> &symb) {

    int sa[N], sb[N];
    syma.write(icompa, sa);
    for(size_t icompb = 0; icompb != symb.get_ncomp(); icompb++) {
        symb.write(icompb, sb);
        bool same = true;
        for(size_t i = 0; i < N; i++) if(sa[i] != sb[i]) same = false;
        if(same) return std::make_pair(true, icompb);
    }
    for(size_t icompb = 0; icompb != symb.get_ncomp(); icompb++) {
        symb.write(icompb, sb);
        bool good = true;
        for(size_t i = 0; i < N; i++) {
            if(!(sa[i] != NS && sb[i] == NS)) good = false;
        }
        if(good) return std::make_pair(false, icompb);
    }
    return std::make_pair(false, size_t(0));
}


template<size_t N, typename T>
size_t ctf_tod_aux_symcomp(const ctf_symmetry<N, T> &syma, size_t icompa,
    const ctf_symmetry<N, T> &symb) {

    return ctf_tod_aux_symcomp_ex(syma, icompa, symb).second;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_AUX_SYMCOMP_H

