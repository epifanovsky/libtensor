#ifndef LIBTENSOR_GEN_BTO_COPY_NZORB_H
#define LIBTENSOR_GEN_BTO_COPY_NZORB_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include "block_list.h"
#include "../gen_block_tensor_i.h"

namespace libtensor {


template<size_t N, typename Traits>
class gen_bto_copy_nzorb : public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta; //!< Block tensor A
    tensor_transf<N, element_type> m_tra; //!< Tensor transformation (A to B)
    symmetry<N, element_type> m_symb; //!< Symmetry of B
    block_list<N> m_blstb; //!< List of non-zero canonical blocks in B

public:
    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param bta First block tensor (A).
        \param btb Second block tensor (B).
        \param symc Symmetry of the result of the contraction (C).
     **/
    gen_bto_copy_nzorb(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        const symmetry<N, element_type> &symb);

    /** \brief Returns the list of non-zero canonical blocks
     **/
    const block_list<N> &get_blst() const {
        return m_blstb;
    }

    /** \brief Builds the list
     **/
    void build();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COPY_NZORB_H
