#ifndef LIBTENSOR_BTO_DIAG_H
#define LIBTENSOR_BTO_DIAG_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_diag.h>

namespace libtensor {


/** \brief Extracts a general diagonal from a block %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \sa gen_bto_diag

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, size_t M, typename T>
class bto_diag :
    public additive_gen_bto<M, typename bto_traits<T>::bti_traits>,
    public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;

private:
    gen_bto_diag<N, M, bto_traits<T>, bto_diag<N, M, T> > m_gbto;

public:
    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param c Scaling factor
     **/
    bto_diag(
        block_tensor_rd_i<N, T> &bta,
        const sequence<N, size_t> &m,
        T c = 1.0) :

        m_gbto(bta, m, tensor_transf<M, T>(permutation<M>(),
                scalar_transf<T>(c))) {

    }

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param p Permutation of result tensor
        \param c Scaling factor
     **/
    bto_diag(
        block_tensor_rd_i<N, T> &bta,
        const sequence<N, size_t> &m,
        const permutation<M> &p,
        T c = 1.0) :

        m_gbto(bta, m, tensor_transf<M, T>(p, scalar_transf<T>(c))) {
    }

    virtual ~bto_diag() { }

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<M> &get_bis() const {

        return m_gbto.get_bis();
    }

    virtual const symmetry<M, T> &get_symmetry() const {

        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<M, T> &get_schedule() const {

        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<M, bti_traits > &out) {

        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<M, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<M, bti_traits > &btb);

    virtual void perform(gen_block_tensor_i<M, bti_traits > &btb,
            const scalar_transf<T> &c);

    virtual void compute_block(
            bool zero,
            const index<M> &ib,
            const tensor_transf<M, T> &trb,
            dense_tensor_wr_i<M, T> &blkb);

    virtual void compute_block(
            const index<M> &ib,
            dense_tensor_wr_i<M, T> &blkb) {

        compute_block(true, ib, tensor_transf<M, T>(), blkb);
    }

    //@}

    /** \brief Convenience wrapper to function
            \c perform(gen_block_tensor_i<M, bti_traits> &, const scalar_transf<T>&)
        \param btb Result block tensor
        \param c Factor
     **/
    void perform(block_tensor_i<M, T> &btb, T c);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_DIAG_H
