#ifndef LIBTENSOR_CTF_BTOD_SET_DIAG_IMPL_H
#define LIBTENSOR_CTF_BTOD_SET_DIAG_IMPL_H

#include <libtensor/ctf_dense_tensor/ctf_tod_set.h>
#include <libtensor/ctf_dense_tensor/ctf_tod_set_diag.h>
#include "../ctf_btod_set_diag.h"

namespace libtensor {


template<size_t N>
const char ctf_btod_set_diag<N>::k_clazz[] = "ctf_btod_set_diag<N>";


template<size_t N>
ctf_btod_set_diag<N>::ctf_btod_set_diag(const sequence<N, size_t> &msk,
    double v) :

    m_gbto(msk, v) {

}


template<size_t N>
ctf_btod_set_diag<N>::ctf_btod_set_diag(double v) :
    m_gbto(sequence<N, size_t>(1), v) {

}


template<size_t N>
void ctf_btod_set_diag<N>::perform(ctf_block_tensor_i<N, double> &bt) {

    m_gbto.perform(bt);
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_DIAG_IMPL_H
