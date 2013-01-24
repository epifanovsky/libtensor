#ifndef LIBTENSOR_CUDA_BTOD_CONTRACT2_H
#define LIBTENSOR_CUDA_BTOD_CONTRACT2_H

#include <libtensor/cuda_block_tensor/cuda_btod_traits.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_contract2.h>

namespace libtensor {




/** \brief Computes the contraction of two block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, size_t K>
class cuda_btod_contract2 :
    public additive_gen_bto<N + M, cuda_btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    typedef typename cuda_btod_traits::bti_traits bti_traits;

private:
    gen_bto_contract2< N, M, K, cuda_btod_traits, cuda_btod_contract2<N, M, K> > m_gbto;

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
    **/
    cuda_btod_contract2(
        const contraction2<N, M, K> &contr,
        cuda_block_tensor_rd_i<NA, double> &bta,
        cuda_block_tensor_rd_i<NB, double> &btb);

    /** \brief Virtual destructor
     **/
    virtual ~cuda_btod_contract2() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    /** \brief Returns the block index space of the result
     **/
    virtual const block_index_space<NC> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N + M, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    virtual const assignment_schedule<N + M, double> &get_schedule() const {

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

    /** \brief Computes the contraction and adds to an block tensor
        \param btc Output tensor.
        \param d Scalar transformation
     **/
    virtual void perform(gen_block_tensor_i<NC, bti_traits> &btc,
        const scalar_transf<double> &d);

    virtual void compute_block(
        bool zero,
        const index<NC> &ic,
        const tensor_transf<NC, double> &trc,
        dense_tensor_wr_i<NC, double> &blkc);

    virtual void compute_block(
        const index<NC> &ic,
        dense_tensor_wr_i<NC, double> &blkc) {

        compute_block(true, ic, tensor_transf<NC, double>(), blkc);
    }

    //@}

    void perform(cuda_block_tensor_i<NC, double> &btc, double d);
};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_BTOD_CONTRACT2_H
