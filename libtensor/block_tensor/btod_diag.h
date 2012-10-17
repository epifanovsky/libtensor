#ifndef LIBTENSOR_BTOD_DIAG_H
#define LIBTENSOR_BTOD_DIAG_H

#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_diag.h>

namespace libtensor {

/** \brief Extracts a general diagonal from a block %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \sa gen_bto_diag

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_diag :
    public additive_gen_bto<N - M + 1, btod_traits::bti_traits>,
    public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_diag<N, M, btod_traits, btod_diag<N, M> > m_gbto;

public:
    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param c Scaling factor
     **/
    btod_diag(
        block_tensor_rd_i<N, double> &bta,
        const mask<N> &m,
        double c = 1.0) :

        m_gbto(bta, m, tensor_transf<N - M + 1, double>(
            permutation<N - M + 1>(), scalar_transf<double>(c))) {

    }

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param p Permutation of result tensor
        \param c Scaling factor
     **/
    btod_diag(
        block_tensor_rd_i<N, double> &bta,
        const mask<N> &m,
        const permutation<N - M + 1> &p,
        double c = 1.0) :

        m_gbto(bta, m, tensor_transf<N - M + 1, double>(
            p, scalar_transf<double>(c))) {
    }

    virtual ~btod_diag() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N - M + 1> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<N - M + 1, double> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N - M + 1, double> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N - M + 1, bti_traits> &out) {

        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb);

    virtual void perform(gen_block_tensor_i<N - M + 1, bti_traits> &btb,
            const scalar_transf<double> &c);

    virtual void compute_block(
            bool zero,
            const index<N - M + 1> &ib,
            const tensor_transf<N - M + 1, double> &trb,
            dense_tensor_wr_i<N - M + 1, double> &blkb);

    //@}

    void perform(block_tensor_i<N - M + 1, double> &btb, double c);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
