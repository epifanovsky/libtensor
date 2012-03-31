#ifndef LIBTENSOR_BTO_DIAG_H
#define LIBTENSOR_BTO_DIAG_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>

namespace libtensor {


/** \brief Extracts a general diagonal from a block %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, typename Traits>
class bto_diag :
    public additive_bto<N - M + 1, typename Traits::additive_bto_traits>,
    public timings< bto_diag<N, M, Traits> > {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensor A
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensora_t;

    //! Type of blocks of a block tensor A
    typedef typename Traits::template block_type<N>::type blocka_t;

    //! Type of block tensor B
    typedef typename Traits::template block_tensor_type<N - M + 1>::type
        block_tensorb_t;

    //! Type of blocks of a block tensor B
    typedef typename Traits::template block_type<N - M + 1>::type blockb_t;

    typedef tensor_transf<N, element_t> tensora_tr_t;

    typedef tensor_transf<N - M + 1, element_t> tensorb_tr_t;

    typedef scalar_transf<element_t> scalar_tr_t;

public:
    static const char *k_clazz; //!< Class name

    static const size_t k_ordera = N; //!< Order of the argument
    static const size_t k_orderb = N - M + 1; //!< Order of the result

private:
    block_tensora_t &m_bta; //!< Input block %tensor
    mask<N> m_msk; //!< Diagonal %mask
    tensorb_tr_t m_tr; //!< Tensor transformation
    block_index_space<k_orderb> m_bis; //!< Block %index space of the result
    symmetry<k_orderb, element_t> m_sym; //!< Symmetry of the result
    assignment_schedule<k_orderb, element_t> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param c Scaling factor
     **/
    bto_diag(block_tensora_t &bta, const mask<N> &m,
        const scalar_tr_t &c = scalar_tr_t());

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param p Permutation of result tensor
        \param c Scaling factor
     **/
    bto_diag(block_tensora_t &bta, const mask<N> &m,
        const permutation<N - M + 1> &p, const scalar_tr_t &c = scalar_tr_t());

    //@}

    //!    \name Implementation of
    //      libtensor::direct_tensor_operation<N - M + 1, double>
    //@{

    virtual const block_index_space<k_orderb> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<k_orderb, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<k_orderb, double> &get_schedule() const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    //@}

    using additive_bto<k_orderb, typename Traits::additive_bto_traits>::perform;

protected:
    virtual void compute_block(bool zero, blockb_t &blk,
        const index<k_orderb> &ib, const tensorb_tr_t &trb,
        const scalar_tr_t &c, cpu_pool &cpus);

private:
    /** \brief Forms the block %index space of the output or throws an
            exception if the input is incorrect.
     **/
    static block_index_space<N - M + 1> mk_bis(
        const block_index_space<N> &bis, const mask<N> &msk);

    /** \brief Sets up the symmetry of the operation result
     **/
    void make_symmetry();

    /** \brief Sets up the assignment schedule for the operation.
     **/
    void make_schedule();

    void compute_block(blockb_t &blk, const index<k_orderb> &ib,
            const tensorb_tr_t &trb, bool zero,
            const scalar_tr_t &c, cpu_pool &cpus);

private:
    bto_diag(const bto_diag<N, M, Traits>&);
    const bto_diag<N, M, Traits> &operator=(const bto_diag<N, M, Traits>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_DIAG_H
