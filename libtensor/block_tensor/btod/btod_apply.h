#ifndef LIBTENSOR_BTOD_APPLY_H
#define LIBTENSOR_BTOD_APPLY_H

#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_apply.h>
#include <libtensor/block_tensor/bto/impl/bto_apply_impl.h>
#include <libtensor/dense_tensor/tod_apply.h>
#include <libtensor/dense_tensor/tod_set.h>

namespace libtensor {


template<size_t N, typename Functor, typename Alloc = std_allocator<double> >
class btod_apply : public bto_apply< N, Functor, btod_traits> {
private:
    typedef bto_apply<N, Functor, btod_traits> bto_apply_type;

public:
    btod_apply(block_tensor_i<N, double> &bta,
            const Functor &fn, double c = 1.0) :
        bto_apply_type(bta, fn, scalar_transf<double>(c)) {
    }

    btod_apply(block_tensor_i<N, double> &bta, const Functor &fn,
            const permutation<N> &p, double c = 1.0) :
        bto_apply_type(bta, fn, p, scalar_transf<double>(c)) {
    }

    virtual ~btod_apply() { }

private:
    btod_apply(const btod_apply&);
    btod_apply &operator=(const btod_apply&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_APPLY_H
