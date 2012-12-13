#ifndef LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_H
#define LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_H

#include "block_list.h"

namespace libtensor {


/** \brief Unfolds a list of non-zero canonical blocks into a list of all
        non-zero blocks
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.

    For every non-zero canonical block, this algorithm expands the full orbit
    and forms a list of all non-zero blocks, both canonical and non-canonical.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_unfold_block_list {
public:
    typedef typename Traits::element_type element_type;

private:
    const symmetry<N, element_type> &m_sym; //!< Symmetry
    const block_list<N> &m_blst; //!< List of non-zero canonical blocks

public:
    gen_bto_unfold_block_list(
        const symmetry<N, element_type> &sym,
        const block_list<N> &blst) :

        m_sym(sym), m_blst(blst)
    { }

    /** \brief Builds a list of all non-zero blocks
     **/
    void build(block_list<N> &blstx);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_BLOCK_LIST_H
