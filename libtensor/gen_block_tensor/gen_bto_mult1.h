#ifndef LIBTENSOR_GEN_BTO_MULT1_H
#define LIBTENSOR_GEN_BTO_MULT1_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Elementwise multiplication of one block tensors with another
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits

    This is a variant to gen_bto_mult. It computes
    \f[
    A_{ij...} =  A_{ij...} + c A_{ij...} \hat{\mathcal{T}}_b B_{ij...}
    \f]
    or
    \f[
    A_{ij...} =  A_{ij...} +
        c A_{ij...} / \left(\hat{\mathcal{T}}_b B_{ij...}\right)
    \f]

    \sa gen_bto_mult

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_mult1 : public timings<Timed>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    //! Type of tensor transformation
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_btb; //!< Tensor B
    tensor_transf_type m_trb; //!< Tensor transformation of B
    bool m_recip; //!< Reciprocal
    scalar_transf<element_type> m_c; //!< Tensor transformation of result

public:
    gen_bto_mult1(
            gen_block_tensor_rd_i<N, bti_traits> &btb,
            const tensor_transf_type &trb,
            bool recip = false,
            const scalar_transf<element_type> &c =
                    scalar_transf<element_type>());

    /** \brief Computes and writes the blocks of the result
        \param zero If false, add the result of the operation to the tensor
        \param bta Output tensor
     **/
    void perform(bool zero, gen_block_tensor_i<N, bti_traits> &bta);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_H
