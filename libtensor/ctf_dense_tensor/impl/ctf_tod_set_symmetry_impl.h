#ifndef LIBTENSOR_CTF_TOD_SET_SYMMETRY_IMPL_H
#define LIBTENSOR_CTF_TOD_SET_SYMMETRY_IMPL_H

#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_set_symmetry.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_set_symmetry<N>::k_clazz[] = "ctf_tod_set_symmetry<N>";


template<size_t N>
void ctf_tod_set_symmetry<N>::perform(bool zero,
    ctf_dense_tensor_i<N, double> &ta) {

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    if(zero) {
        ca.reset_symmetry(m_sym);
    } else {
        ca.adjust_symmetry(m_sym);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_SYMMETRY_IMPL_H
