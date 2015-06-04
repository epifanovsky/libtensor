#ifndef LIBTENSOR_CTF_SYMMETRY_IMPL_H
#define LIBTENSOR_CTF_SYMMETRY_IMPL_H

#include "../ctf.h"
#include "../ctf_symmetry.h"

namespace libtensor {


template<size_t N, typename T>
ctf_symmetry<N, T>::ctf_symmetry() {

    sequence<N, unsigned> grp, sym;
    for(size_t i = 0; i < N; i++) {
        grp[i] = i;
        sym[i] = 0;
    }
    add_component(grp, sym);
}


template<size_t N, typename T>
ctf_symmetry<N, T>::ctf_symmetry(const sequence<N, unsigned> &grp,
    const sequence<N, unsigned> &sym, bool jilk) {

    if(jilk) {
        sequence<N, unsigned> grp, sym;
        for(size_t i = 0; i < N/2; i++) grp[i] = 0;
        for(size_t i = N/2; i < N; i++) grp[i] = 1;
        sym[0] = 0; sym[1] = 0;
        m_sym.push_back(std::make_pair(grp, sym));
        sym[0] = 1; sym[1] = 1;
        m_sym.push_back(std::make_pair(grp, sym));
    } else {
        m_sym.push_back(std::make_pair(grp, sym));
    }
}


template<size_t N, typename T>
void ctf_symmetry<N, T>::add_component(const sequence<N, unsigned> &grp,
    const sequence<N, unsigned> &sym) {

    m_sym.push_back(std::make_pair(grp, sym));
}


template<size_t N, typename T>
bool ctf_symmetry<N, T>::is_subgroup(const ctf_symmetry<N, T> &other) const {

    return true;
}


template<size_t N, typename T>
void ctf_symmetry<N, T>::permute(const permutation<N> &perm) {

    for(size_t i = 0; i < m_sym.size(); i++) {
        perm.apply(m_sym[i].first);
    }
}


namespace {
    static bool g_use_ctf_symmetry = true;
} // unnamed namespace


template<size_t N, typename T>
void ctf_symmetry<N, T>::write(size_t icomp, int (&sym)[N]) const {

    if(g_use_ctf_symmetry) {

        const sequence<N, unsigned> &grp0 = m_sym[icomp].first;
        const sequence<N, unsigned> &sym0 = m_sym[icomp].second;
        sequence<N, unsigned> grp(0);
        for(size_t i = 0; i < N; i++) grp[i] = grp0[N - 1 - i];
        size_t i = 0;
        size_t symidx = 0;
        while(i < N) {
            unsigned g = grp[i];
            size_t i0 = i;
            while(i < N && grp[i] == g) i++;
            int symasym = sym0[g] == 0 ? SY : AS;
            sym[i0] = NS;
            for(size_t j = i0 + 1; j < i; j++) {
                sym[j - 1] = symasym;
                sym[j] = NS;
                symidx++;
            }
        }

    } else {

        for(size_t i = 0; i < N; i++) sym[i] = NS;

    }
}


namespace {

unsigned long symmetry_factor(size_t N, int *sym) {

    unsigned long f = 1;
    for(size_t i = 0, j = 0; i < N; i++) {
        if(sym[i] == SY || sym[i] == AS) {
            j++;
            continue;
        }
        if(sym[i] == NS) {
            j++;
            for(size_t k = 2; k <= j; k++) f *= k;
            j = 0;
        }
    }
    return f;
}

} // unnamed namespace


template<size_t N, typename T>
T ctf_symmetry<N, T>::symconv_factor(const ctf_symmetry<N, T> &syma,
    size_t icompa, const ctf_symmetry<N, T> &symb, size_t icompb) {

    int sa[N], sb[N];
    syma.write(icompa, sa);
    symb.write(icompb, sb);

    for(size_t i = 0; i < N; i++) {
        int ga = sa[i], gb = sb[i];
        if(ga == NS || gb == NS) continue;
        if(ga != gb) return 0.0;
    }

    for(size_t i = 0; i < N; i++) {
        if(sb[i] == NS) sa[i] = NS;
    }
    unsigned long fa = symmetry_factor(N, sa);
    unsigned long fb = symmetry_factor(N, sb);

    //  It might be tempting to optimize the expression below, but don't
    //  do it to avoid errors. It is exactly right as written.
    return T(1.0 / double(fb / fa));
}


template<size_t N, typename T>
void ctf_symmetry<N, T>::use_ctf_symmetry(bool use) {

    g_use_ctf_symmetry = use;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_IMPL_H
