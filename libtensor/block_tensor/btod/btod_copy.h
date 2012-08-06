#ifndef LIBTENSOR_BTOD_COPY_H
#define LIBTENSOR_BTOD_COPY_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_copy.h>

namespace libtensor {


template<size_t N>
class btod_copy : public bto_copy<N, btod_traits> {
public:
    typedef bto_copy<N, btod_traits> bto_copy_type;
    typedef typename bto_copy_type::scalar_transf_type scalar_transf_type;

public:
    btod_copy(block_tensor_i<N, double> &bta, double c = 1.0) :
        bto_copy_type(bta, scalar_transf_type(c)) {
    }

    btod_copy(block_tensor_i<N, double> &bta, const permutation<N> &p,
        double c = 1.0) :
        bto_copy_type(bta, p, scalar_transf_type(c)) {
    }

    virtual ~btod_copy() { }

private:
    btod_copy(const btod_copy&);
    btod_copy &operator=(const btod_copy&);

};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_COPY_H
