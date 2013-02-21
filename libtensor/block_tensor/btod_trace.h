#ifndef LIBTENSOR_BTOD_TRACE_H
#define LIBTENSOR_BTOD_TRACE_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_trace.h>

namespace libtensor {


/** \brief Computes the trace of a matricized block %tensor
    \tparam N Tensor diagonal order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_trace : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

    enum { NA = 2 * N };

private:
    gen_bto_trace< N, btod_traits, btod_trace<N> > m_gbto;

public:
    btod_trace(block_tensor_rd_i<NA, double> &bta) :
        m_gbto(bta, permutation<NA>()) {
    }

    btod_trace(block_tensor_rd_i<NA, double> &bta,
            const permutation<NA> &perm) :
        m_gbto(bta, perm) {
    }

    double calculate() {
        return m_gbto.calculate();
    }
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_TRACE_H
