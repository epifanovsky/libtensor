#ifndef LIBTENSOR_BTO_EWMULT2_H
#define LIBTENSOR_BTO_EWMULT2_H

#include <libtensor/block_tensor/bto_traits.h>
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

    \sa tod_ewmult2, bto_contract2

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_ewmult2 :
    public additive_gen_bto<N + M + K, typename bto_traits<T>::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M + K //!< Order of result (C)
    };

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;
    typedef tensor_transf<NC, T> tensor_transf_type;

private:
    gen_bto_ewmult2< N, M, K, bto_traits<T>, bto_ewmult2<N, M, K, T> > m_gbto;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param btb Second argument (B).
        \param d Scaling coefficient.
     **/
    bto_ewmult2(block_tensor_rd_i<NA, T> &bta,
        block_tensor_rd_i<NB, T> &btb, T d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param perma Permutation of A.
        \param btb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param d Scaling coefficient.
     **/
    bto_ewmult2(block_tensor_rd_i<NA, T> &bta,
        const permutation<NA> &perma,
        block_tensor_rd_i<NB, T> &btb,
        const permutation<NB> &permb,
        const permutation<NC> &permc, T d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param tra Tensor transformation of A.
        \param btb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    bto_ewmult2(block_tensor_rd_i<NA, T> &bta,
        const tensor_transf<NA, T> &tra,
        block_tensor_rd_i<NB, T> &btb,
        const tensor_transf<NB, T> &trb,
        const tensor_transf_type &trc = tensor_transf_type());

    /** \brief Virtual destructor
     **/
    virtual ~bto_ewmult2() { }

    //@}


    //!    \name Implementation of direct_gen_bto<N + M + K, bti_traits>
    //@{

    virtual const block_index_space<N + M + K> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N + M + K, T> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N + M + K, T> &get_schedule()
        const {
        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<NC, bti_traits> &out);

    //@}


    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc);

    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc,
            const scalar_transf<T> &d);

    virtual void compute_block(
            bool zero,
            const index<NC> &i,
            const tensor_transf<NC, T> &tr,
            dense_tensor_wr_i<NC, T> &blk);

    virtual void compute_block(
            const index<NC> &ic,
            dense_tensor_wr_i<NC, T> &blkc) {

        compute_block(true, ic, tensor_transf<NC, T>(), blkc);
    }

    //@}

    void perform(block_tensor_i<NC, T> &btc, T d);

};

template<size_t N, size_t M, size_t K>
using btod_ewmult2 = bto_ewmult2<N, M, K, double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_EWMULT2_H
