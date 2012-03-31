#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include "scalar_transf_double.h"
#include <libtensor/block_tensor/bto/bto_copy.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


struct btod_copy_traits : public bto_traits<double> {
};


template<size_t N>
class btod_copy : public bto_copy<N, btod_copy_traits> {
private:
    typedef bto_copy<N, btod_copy_traits> bto_copy_t;
    typedef typename bto_copy_t::scalar_tr_t scalar_tr_t;

public:
    btod_copy(block_tensor_i<N, double> &bta, double c = 1.0) :
        bto_copy_t(bta, scalar_tr_t(c)) {
    }

    btod_copy(block_tensor_i<N, double> &bta,
            const permutation<N> &p, double c = 1.0) :
        bto_copy_t(bta, p, scalar_tr_t(c)) {
    }

    virtual ~btod_copy() { }

private:
    btod_copy(const btod_copy<N>&);
    btod_copy<N> &operator=(const btod_copy<N>&);
};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_COPY_H
