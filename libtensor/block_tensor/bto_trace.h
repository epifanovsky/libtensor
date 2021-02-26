#ifndef LIBTENSOR_BTO_TRACE_H
#define LIBTENSOR_BTO_TRACE_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_trace.h>

namespace libtensor {


/** \brief Computes the trace of a matricized block %tensor
    \tparam N Tensor diagonal order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_trace : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

    enum { NA = 2 * N };

private:
    gen_bto_trace< N, bto_traits<T>, bto_trace<N, T> > m_gbto;

public:
    bto_trace(block_tensor_rd_i<NA, T> &bta) :
        m_gbto(bta, permutation<NA>()) {
    }

    bto_trace(block_tensor_rd_i<NA, T> &bta,
            const permutation<NA> &perm) :
        m_gbto(bta, perm) {
    }

    T calculate() {
        return m_gbto.calculate();
    }
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_TRACE_H
