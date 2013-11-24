#ifndef LIBTENSOR_GEN_BTO_SCALE_H
#define LIBTENSOR_GEN_BTO_SCALE_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Apply a scalar transformation to a block %tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template to_add_type<N>::type -- Type of tensor operation to_copy
    - \c template to_copy_type<N>::type -- Type of tensor operation to_add
    - \c template to_random_type<N>::type -- Type of tensor operation to_random

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_scale : public timings<Timed>, public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

public:
    static const char k_clazz[]; //!< Class name

private:
    gen_block_tensor_i<N, bti_traits> &m_bt; //!< Block %tensor
    scalar_transf<element_type> m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param bt Block %tensor.
        \param c Scaling coefficient.
     **/
    gen_bto_scale(
        gen_block_tensor_i<N, bti_traits> &bt,
        const scalar_transf<element_type> &c) :

        m_bt(bt), m_c(c)
    { }

    /** \brief Performs the operation
     **/
    void perform();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SCALE_H
