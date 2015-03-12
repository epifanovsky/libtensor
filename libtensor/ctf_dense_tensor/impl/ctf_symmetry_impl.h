#ifndef LIBTENSOR_CTF_SYMMETRY_IMPL_H
#define LIBTENSOR_CTF_SYMMETRY_IMPL_H

#include "../ctf.h"
#include "../ctf_symmetry.h"

namespace libtensor {


template<size_t N, typename T>
ctf_symmetry<N, T>::ctf_symmetry() {

}


template<size_t N, typename T>
void ctf_symmetry<N, T>::build(const transf_list<N, T> &trl) {

}


template<size_t N, typename T>
void ctf_symmetry<N, T>::permute(const permutation<N> &perm) {

}


template<size_t N, typename T>
void ctf_symmetry<N, T>::write(int (&sym)[N]) const {

    for(size_t i = 0; i < N; i++) sym[i] = NS;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_SYMMETRY_IMPL_H
