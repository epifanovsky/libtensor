#ifndef LIBTENSOR_BTO_TRACE_H
#define LIBTENSOR_BTO_TRACE_H

#include <libtensor/defs.h>
#include <libtensor/timings.h>
#include <libtensor/core/permutation.h>

namespace libtensor {


/** \brief Computes the trace of a matricized block %tensor
    \tparam N Tensor diagonal order.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class bto_trace : public timings< bto_trace<N, Traits> > {
public:
    static const char *k_clazz; //!< Class name

public:
    static const size_t k_ordera = 2 * N; //!< Order of the argument

public:
    typedef typename Traits::template block_tensor_type<k_ordera>::type
        block_tensor_t;

private:
    block_tensor_t &m_bta; //!< Input block %tensor
    permutation<k_ordera> m_perm; //!< Permutation of the %tensor

public:
    bto_trace(block_tensor_t &bta);

    bto_trace(block_tensor_t &bta,
        const permutation<k_ordera> &perm);

    double calculate();

private:
    bto_trace(const bto_trace<N, Traits>&);
    const bto_trace<N, Traits> &operator=(const bto_trace<N, Traits>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_TRACE_H
