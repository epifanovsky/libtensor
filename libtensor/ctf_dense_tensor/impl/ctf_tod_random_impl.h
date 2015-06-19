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

    ctf_dense_tensor_ctrl<N, double> ca(ta);
    CTF::Tensor<double> &dta = ca.req_ctf_tensor();

    int64_t np, npmax, *idx;
    double *p;
    dta.read_local(&np, &idx, &p);
    MPI_Allreduce(&np, &npmax, 1, MPI_INT64_T, MPI_MAX, ctf::get_world().comm);

    size_t sz = npmax;
    double *buf = new double[sz];

    try {

        linalg::rng_set_i_x(0, sz, buf, 1, m_c);

        if(zero) {
            for(size_t i = 0; i < np; i++) p[i] = buf[i];
        } else {
            for(size_t i = 0; i < np; i++) p[i] += buf[i];
        }
        dta.write(np, idx, p);

        free(idx);
        free(p);

    } catch(...) {
        delete [] buf;
        throw;
    }

    delete [] buf;
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_RANDOM_IMPL_H
