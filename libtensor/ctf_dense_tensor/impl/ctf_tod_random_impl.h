#ifndef LIBTENSOR_CTF_TOD_RANDOM_IMPL_H
#define LIBTENSOR_CTF_TOD_RANDOM_IMPL_H

#include <libtensor/linalg/linalg.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_tod_random.h"

namespace libtensor {


template<size_t N>
const char ctf_tod_random<N>::k_clazz[] = "ctf_tod_random<N>";


template<size_t N>
ctf_tod_random<N>::ctf_tod_random(const scalar_transf<double> &c) :
    m_c(c.get_coeff()) {

}


template<size_t N>
ctf_tod_random<N>::ctf_tod_random(double c) : m_c(c) {

}


template<size_t N>
void ctf_tod_random<N>::perform(bool zero, ctf_dense_tensor_i<N, double> &ta) {

    size_t totsz = ta.get_dims().get_size();

    unsigned iproc = ctf::get_rank(), nproc = ctf::get_size();
    size_t off = iproc * totsz / nproc;
    size_t sz = std::min((iproc + 1) * totsz / nproc, totsz) - off;

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    tCTF_Tensor<double> &dta = ca.req_ctf_tensor();

    const size_t bufsz = 1024 * 1024;
    std::vector<long_int> idx(bufsz);
    std::vector<double> data(bufsz);
    for(size_t i = 0; i < sz; i += bufsz) {
        size_t n = std::min(sz - i, bufsz);
        for(size_t j = 0; j < n; j++) idx[j] = long_int(off + i + j);
        if(zero) {
            linalg::rng_set_i_x(0, n, &data[0], 1, m_c);
        } else {
            dta.read(n, &idx[0], &data[0]);
            linalg::rng_add_i_x(0, n, &data[0], 1, m_c);
        }
        dta.write(n, &idx[0], &data[0]);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_RANDOM_IMPL_H
