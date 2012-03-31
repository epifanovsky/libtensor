#ifndef LIBTENSOR_BTOD_APPLY_H
#define LIBTENSOR_BTOD_APPLY_H

#include <libtensor/core/allocator.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/dense_tensor/tod_apply.h>
#include <libtensor/block_tensor/bto/bto_apply.h>
#include <libtensor/block_tensor/bto/impl/bto_apply_impl.h>

namespace libtensor {


template<typename Functor, typename Alloc>
struct btod_apply_traits : public bto_traits<double> {

    typedef bto_traits<double> additive_bto_traits;

    template<size_t N> struct tensor_type {
        typedef dense_tensor<N, double, Alloc> type;
    };

    typedef Functor functor_type;

    template<size_t N> struct to_apply_type {
        typedef tod_apply<N, Functor> type;
    };

    static double zero() { return 0.0; }
};


template<size_t N, typename Functor, typename Alloc = std_allocator<double> >
class btod_apply : public bto_apply<N, btod_apply_traits<Functor, Alloc> > {
private:
    typedef bto_apply<N, btod_apply_traits<Functor, Alloc> > bto_apply_t;
    typedef typename bto_apply_t::scalar_tr_t scalar_tr_t;

public:
    btod_apply(block_tensor_i<N, double> &bta,
            const Functor &fn, double c = 1.0) :
        bto_apply_t(bta, fn, scalar_tr_t(c)) {
    }

    btod_apply(block_tensor_i<N, double> &bta, const Functor &fn,
            const permutation<N> &p, double c = 1.0) :
        bto_apply_t(bta, fn, p, scalar_tr_t(c)) {
    }

private:
    btod_apply(const btod_apply<N, Functor, Alloc>&);
    btod_apply<N, Functor, Alloc> &
    operator=(const btod_apply<N, Functor, Alloc>&);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_APPLY_H
