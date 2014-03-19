#ifndef LIBTENSOR_CTF_BTOD_TRACE_H
#define LIBTENSOR_CTF_BTOD_TRACE_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_trace.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Computes the trace of a matricized distributed CTF block %tensor
    \tparam N Tensor diagonal order.

    \sa gen_bto_trace, btod_trace

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_trace : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

    enum {
        NA = 2 * N
    };

private:
    gen_bto_trace< N, ctf_btod_traits, ctf_btod_trace<N> > m_gbto;

public:
    ctf_btod_trace(
        ctf_block_tensor_rd_i<NA, double> &bta) :

        m_gbto(bta, permutation<NA>()) {
    }

    ctf_btod_trace(
        ctf_block_tensor_rd_i<NA, double> &bta,
        const permutation<NA> &perm) :

        m_gbto(bta, perm) {
    }

    double calculate() {
        return m_gbto.calculate();
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_TRACE_H
