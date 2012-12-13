#ifndef LIBTENSOR_GEN_BTO_TRACE_H
#define LIBTENSOR_GEN_BTO_TRACE_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the trace of a matricized block %tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_trace_type<N>::type -- Type of tensor operation to_trace

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_trace : public timings<Timed>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = 2 * N
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< Input block %tensor
    permutation<NA> m_perm; //!< Permutation of the %tensor

public:
    gen_bto_trace(gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const permutation<NA> &perm);

    element_type calculate();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_TRACE_H
