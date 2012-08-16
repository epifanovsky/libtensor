#ifndef LIBTENSOR_BTOD_TRACE_H
#define LIBTENSOR_BTOD_TRACE_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_trace.h>
#include <libtensor/dense_tensor/tod_trace.h>

namespace libtensor {


struct btod_trace_traits : public bto_traits<double> {

    template<size_t N> struct to_trace_type {
        typedef tod_trace<N> type;
    };

};

/** \brief Computes the trace of a matricized block %tensor
    \tparam N Tensor diagonal order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_trace : public bto_trace<N, btod_trace_traits> {
public:
    typedef bto_trace<N, btod_trace_traits> bto_trace_t;

public:
    btod_trace(block_tensor_i<bto_trace_t::k_ordera, double> &bta) :
        bto_trace_t(bta) {
    }

    btod_trace(block_tensor_i<bto_trace_t::k_ordera, double> &bta,
            const permutation<bto_trace_t::k_ordera> &perm) :
        bto_trace_t(bta, perm) {
    }

private:
    btod_trace(const btod_trace<N>&);
    const btod_trace<N> &operator=(const btod_trace<N>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRACE_H
