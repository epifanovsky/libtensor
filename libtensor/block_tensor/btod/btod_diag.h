#ifndef LIBTENSOR_BTOD_DIAG_H
#define LIBTENSOR_BTOD_DIAG_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_diag.h>

namespace libtensor {


template<size_t N, size_t M>
class btod_diag : public bto_diag<N, M, btod_traits> {
private:
    typedef bto_diag<N, M, btod_traits> bto_diag_t;
    typedef typename bto_diag_t::scalar_tr_t scalar_tr_t;

public:
    btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
            double c = 1.0) : bto_diag_t(bta, m, scalar_tr_t(c)) {
    }

    btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
            const permutation<N - M + 1> &p, double c = 1.0) :
        bto_diag_t(bta, m, p, scalar_tr_t(c)) {
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
