#ifndef LIBTENSOR_BTO_CONTRACT2_XM_H
#define LIBTENSOR_BTO_CONTRACT2_XM_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2.h>

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
struct bto_contract2_xm_clazz {
    static const char k_clazz[];
};


/** \brief Computes the contraction of two block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_xm :
    public additive_gen_bto<N + M, typename bto_traits<T>::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    //! Block tensor interface traits
    typedef typename bto_traits<T>::bti_traits bti_traits;

    //! Type of tensor elements
    typedef typename bto_traits<T>::bti_traits::element_type element_type;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First argument (A)
    scalar_transf<element_type> m_ka; //!< Scalar transform of A.
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second argument (B)
    scalar_transf<element_type> m_kb; //!< Scalar transform of B.
    scalar_transf<element_type> m_kc; //!< Scalar transform of the result.

    gen_bto_contract2_sym<N, M, K, bto_traits<T> >
            m_symc; //!< Symmetry of the result
    gen_bto_contract2< N, M, K, bto_traits<T>, bto_contract2_xm<N, M, K, T> >
            m_gbto;

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
    **/
    bto_contract2_xm(
        const contraction2<N, M, K> &contr,
        block_tensor_rd_i<NA, T> &bta,
        block_tensor_rd_i<NB, T> &btb);

    /** \brief Initializes the contraction operation with scaling coefficients
        \param contr Contraction.
        \param bta Block tensor A (first argument).
        \param ka Scalar for A.
        \param btb Block tensor B (second argument).
        \param kb Scalar for B.
        \param kc Scalar for result.
    **/
    bto_contract2_xm(
        const contraction2<N, M, K> &contr,
        block_tensor_rd_i<NA, T> &bta,
        T ka,
        block_tensor_rd_i<NB, T> &btb,
        T kb,
        T kc);

    /** \brief Virtual destructor
     **/
    virtual ~bto_contract2_xm() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    /** \brief Returns the block index space of the result
     **/
    virtual const block_index_space<NC> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N + M, T> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    virtual const assignment_schedule<N + M, T> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the contraction into an output stream
     **/
    virtual void perform(gen_block_stream_i<NC, bti_traits> &out);

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    /** \brief Computes the contraction into an output block tensor
     **/
    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc);

    virtual void partial(gen_block_tensor_i<NC, bti_traits> &,
        const std::vector<size_t> &);


    /** \brief Computes the contraction and adds to an block tensor
        \param btc Output tensor.
        \param d Scalar transformation
     **/
    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc,
        const scalar_transf<T> &d);

    virtual void compute_block(
        bool zero,
        const index<NC> &ic,
        const tensor_transf<NC, T> &trc,
        dense_tensor_wr_i<NC, T> &blkc);

    virtual void compute_block(
        const index<NC> &ic,
        dense_tensor_wr_i<NC, T> &blkc) {

        compute_block(true, ic, tensor_transf<NC, T>(), blkc);
    }

    //@}

    void perform(block_tensor_i<NC, T> &btc, T d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_XM_H
