#ifndef LIBTENSOR_BTOD_EWMULT2_H
#define LIBTENSOR_BTOD_EWMULT2_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_ewmult2.h>

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
    public additive_gen_bto<N + M + K, btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M + K //!< Order of result (C)
    };

public:
    typedef typename btod_traits::bti_traits bti_traits;
    typedef tensor_transf<NC, double> tensor_transf_type;

private:
    gen_bto_ewmult2< N, M, K, btod_traits, btod_ewmult2<N, M, K> > m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param btb Second argument (B).
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_rd_i<NA, double> &bta,
        block_tensor_rd_i<NB, double> &btb, double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param perma Permutation of A.
        \param btb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_rd_i<NA, double> &bta,
        const permutation<NA> &perma,
        block_tensor_rd_i<NB, double> &btb,
        const permutation<NB> &permb,
        const permutation<NC> &permc, double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param tra Tensor transformation of A.
        \param btb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    btod_ewmult2(block_tensor_rd_i<NA, double> &bta,
        const tensor_transf<NA, double> &tra,
        block_tensor_rd_i<NB, double> &btb,
        const tensor_transf<NB, double> &trb,
        const tensor_transf_type &trc = tensor_transf_type());

    /** \brief Virtual destructor
     **/
    virtual ~btod_ewmult2() { }

    //@}


    //!    \name Implementation of direct_gen_bto<N + M + K, bti_traits>
    //@{

    virtual const block_index_space<N + M + K> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N + M + K, double> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N + M + K, double> &get_schedule()
        const {
        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<NC, bti_traits> &out);

    //@}


    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc);

    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc,
            const scalar_transf<double> &d);

    virtual void compute_block(
            bool zero,
            const index<NC> &i,
            const tensor_transf<NC, double> &tr,
            dense_tensor_wr_i<NC, double> &blk);

    virtual void compute_block(
            const index<NC> &ic,
            dense_tensor_wr_i<NC, double> &blkc) {

        compute_block(true, ic, tensor_transf<NC, double>(), blkc);
    }

    //@}

    void perform(block_tensor_i<NC, double> &btc, double d);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_H
