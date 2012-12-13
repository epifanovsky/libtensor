#ifndef LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_H
#define LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_H

#include <list>
#include <libtensor/core/orbit_list.h>
#include "gen_block_stream_i.h"

namespace libtensor {


/** \brief Symmetrizes blocks into a target stream (auxiliary operation)
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits structure.

    This auxiliary block tensor operation acts as a filter that accepts blocks
    and symmetrizes them into a target stream. For each input block,
    the corresponding canonical block in the target symmetry is found, and
    the input block is relayed with the appropriate transformation.

    Because the symmetrization operation involves combining multiple
    contributions into one target block, the output stream must process
    blocks under addition.

    \sa gen_block_stream_i, gen_bto_aux_add

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_aux_symmetrize :
    public gen_block_stream_i<N, typename Traits::bti_traits> {

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

    //! Type of symmetry
    typedef symmetry<N, element_type> symmetry_type;

    //! Type of tensor transformation
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    symmetry_type m_syma; //!< Initial symmetry
    symmetry_type m_symb; //!< Target (symmetrized) symmetry
    std::list<tensor_transf_type> m_trlst; //!< List of transformations
    gen_block_stream_i<N, bti_traits> &m_out; //!< Output stream
    bool m_open; //!< Open state

public:
    /** \brief Constructs the operation
        \brief syma Initial symmetry.
        \brief symb Target symmetry.
        \brief out Output stream.
     **/
    gen_bto_aux_symmetrize(
        const symmetry_type &syma,
        const symmetry_type &symb,
        gen_block_stream_i<N, bti_traits> &out);

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_aux_symmetrize();

    /** \brief Add a transformation to the symmetrizer
     **/
    void add_transf(const tensor_transf_type &tr);

    /** \brief Implements bto_stream_i::open(). Prepares the symmetrization
            operation
     **/
    virtual void open();

    /** \brief Implements bto_stream_i::close()
     **/
    virtual void close();

    /** \brief Implements bto_stream_i::put(). Relays transformed blocks to
            the output stream
     **/
    virtual void put(
        const index<N> &idx,
        rd_block_type &blk,
        const tensor_transf_type &tr);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_SYMMETRIZE_H
