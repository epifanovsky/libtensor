#ifndef LIBTENSOR_BTOD_CONTRACT3_H
#define LIBTENSOR_BTOD_CONTRACT3_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_contract3.h>
#include <libtensor/block_tensor/btod_traits.h>

namespace libtensor {


/** \brief Contracts a train of three tensors
    \tparam N1 Order of first tensor less first contraction degree.
    \tparam N2 Order of second tensor less total contraction degree.
    \tparam N3 Order of third tensor less second contraction degree.
    \tparam K1 First contraction degree.
    \tparam K2 Second contraction degree.

    This algorithm computes the contraction of three linearly connected tensors.

    The contraction is performed as follows. The first tensor is contracted
    with the second tensor to form an intermediate, which is then contracted
    with the third tensor to yield the final result.

    The formation of the intermediate is done in batches:
    \f[
        ABC = A(B_1 + B_2 + \dots + B_n)C = \sum_{i=1}^n (AB_i)C \qquad
        B = \sum_{i=1}^n B_i
    \f]

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
class btod_contract3 :
    public additive_gen_bto<N1 + N2 + N3, btod_traits::bti_traits>,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_contract3<N1, N2, N3, K1, K2, btod_traits,
        btod_contract3<N1, N2, N3, K1, K2> > m_gbto;

public:
    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param btc Third tensor argument (C).
     **/
    btod_contract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_rd_i<N1 + K1, double> &bta,
        block_tensor_rd_i<N2 + K1 + K2, double> &btb,
        block_tensor_rd_i<N3 + K2, double> &btc);

    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param btc Third tensor argument (C).
        \param kd Scaling coefficient.
     **/
    btod_contract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_rd_i<N1 + K1, double> &bta,
        block_tensor_rd_i<N2 + K1 + K2, double> &btb,
        block_tensor_rd_i<N3 + K2, double> &btc,
        double kd);

    /** \brief Virtual destructor
     **/
    virtual ~btod_contract3() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    /** \brief Returns the block index space of the result
     **/
    virtual const block_index_space<N1 + N2 + N3> &get_bis() const {

        return m_gbto.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    virtual const symmetry<N1 + N2 + N3, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    virtual const assignment_schedule<N1 + N2 + N3, double> &get_schedule()
            const {

        return m_gbto.get_schedule();
    }

    /** \brief Computes the contraction into an output stream
     **/
    virtual void perform(gen_block_stream_i<N1 + N2 + N3, bti_traits> &out);

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    /** \brief Computes the contraction into an output block tensor
     **/
    virtual void perform(gen_block_tensor_i<N1 + N2 + N3, bti_traits> &btd);

    /** \brief Computes the contraction and adds to an block tensor
        \param btd Output tensor.
        \param k Scalar transformation
     **/
    virtual void perform(gen_block_tensor_i<N1 + N2 + N3, bti_traits> &btd,
        const scalar_transf<double> &k);

    virtual void compute_block(
        bool zero,
        const index<N1 + N2 + N3> &id,
        const tensor_transf<N1 + N2 + N3, double> &trd,
        dense_tensor_wr_i<N1 + N2 + N3, double> &blkd);

    virtual void compute_block(
        const index<N1 + N2 + N3> &id,
        dense_tensor_wr_i<N1 + N2 + N3, double> &blkd) {

        compute_block(true, id, tensor_transf<N1 + N2 + N3, double>(), blkd);
    }

    //@}

    /** \brief Computes the contraction and adds to an output block tensor
     **/
    void perform(block_tensor_i<N1 + N2 + N3, double> &btd, double k);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT3_H

