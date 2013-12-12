#ifndef LIBTENSOR_GEN_BTO_AUX_DOTPROD_H
#define LIBTENSOR_GEN_BTO_AUX_DOTPROD_H

#include <libutil/threads/mutex.h>
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the dot product between a block tensor and a streaming
        block tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_aux_dotprod :
    public gen_block_stream_i<N, typename Traits::bti_traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of tensor transformation of result
    typedef tensor_transf<N, element_type> tensor_transf_type;

public:
    static const char k_clazz[]; //!< Class name

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta; //!< First argument (A)
    const tensor_transf_type &m_tra; //!< Transformation of A
    block_index_space<N> m_bisb; //!< Block index space of second argument (B)
    symmetry<N, element_type> m_symb; //!< Symmetry of B
    symmetry<N, element_type> m_symc; //!< Symmetry of A*B
    element_type m_d; //!< Dot product
    libutil::mutex m_mtx; //!< Mutex

public:
    /** \brief Initializes the operation
     **/
    gen_bto_aux_dotprod(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf_type &tra,
        const symmetry<N, element_type> &symb);

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_aux_dotprod();

    /** \brief Implements gen_block_stream_i::open().
     **/
    virtual void open();

    /** \brief Implements gen_block_stream_i::close()
     **/
    virtual void close();

    /** \brief Implements gen_block_stream_i::put(). Accepts blocks of B and
            dots them into blocks of A
     **/
    virtual void put(
        const index<N> &idx,
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr);

    /** \brief Returns the accumulated dot product
     **/
    const element_type &get_d() const {
        return m_d;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_DOTPROD_H
