#ifndef LIBTENSOR_BTOD_EWMULT2_H
#define LIBTENSOR_BTOD_EWMULT2_H

#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Generalized element-wise (Hadamard) product of two block tensors
    \tparam N Order of first argument (A) less the number of shared indexes.
    \tparam M Order of second argument (B) less the number of shared
        indexes.
    \tparam K Number of shared indexes.

    This operation computes the element-wise product of two block tensor.
    Refer to tod_ewmult2<N, M, K> for setup info.

    Both arguments and result must agree on their block index spaces,
    otherwise the constructor and perform() will raise
    bad_block_index_space.

    \sa tod_ewmult2, btod_contract2

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_ewmult2 :
    public additive_bto<N + M + K, btod_traits>,
    public timings< btod_ewmult2<N, M, K> > {

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M + K //!< Order of result (C)
    };

private:
    block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
    permutation<k_ordera> m_perma; //!< Permutation of first argument (A)
    block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
    permutation<k_orderb> m_permb; //!< Permutation of second argument (B)
    permutation<k_orderc> m_permc; //!< Permutation of result (C)
    double m_d; //!< Scaling coefficient
    block_index_space<k_orderc> m_bisc; //!< Block index space of result
    symmetry<k_orderc, double> m_symc; //!< Symmetry of result
    assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param btb Second argument (B).
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
        block_tensor_i<k_orderb, double> &btb, double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param perma Permutation of A.
        \param btb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param recip Reciprocal flag.
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
        const permutation<k_ordera> &perma,
        block_tensor_i<k_orderb, double> &btb,
        const permutation<k_orderb> &permb,
        const permutation<k_orderc> &permc, double d = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_ewmult2();

    //@}


    //!    \name Implementation of
    //!        direct_block_tensor_operation<N + M + K, double>
    //@{

    virtual const block_index_space<N + M + K> &get_bis() const {
        return m_bisc;
    }

    virtual const symmetry<N + M + K, double> &get_symmetry() const {
        return m_symc;
    }

    virtual const assignment_schedule<N + M + K, double> &get_schedule()
        const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    virtual void perform(bto_stream_i<N + M + K, btod_traits> &out);
    virtual void perform(block_tensor_i<N + M + K, double> &btc);
    virtual void perform(block_tensor_i<N + M + K, double> &btc,
        const double &d);

    //@}

    using additive_bto<N + M + K, btod_traits>::compute_block;
    virtual void compute_block(bool zero,
        dense_tensor_i<k_orderc, double> &blk, const index<k_orderc> &i,
        const tensor_transf<k_orderc, double> &tr, const double &c);

private:
    /** \brief Computes the block index space of the result block tensor
     **/
    static block_index_space<N + M + K> make_bisc(
        const block_index_space<k_ordera> &bisa,
        const permutation<k_ordera> &perma,
        const block_index_space<k_orderb> &bisb,
        const permutation<k_orderb> &permb,
        const permutation<k_orderc> &permc);

    /** \brief Computes the symmetry of the result block tensor
     **/
    void make_symc();

    /** \brief Prepares the assignment schedule
     **/
    void make_schedule();

    /** \brief Computes the given block of the result
     **/
    void compute_block_impl(dense_tensor_i<k_orderc, double> &blk,
        const index<k_orderc> &bidx, const tensor_transf<k_orderc, double> &tr,
        bool zero, double d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_H
