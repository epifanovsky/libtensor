#ifndef LIBTENSOR_BTO_MULT_H
#define LIBTENSOR_BTO_MULT_H

#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_mult.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_mult :
    public additive_gen_bto<N, typename bto_traits<T>::bti_traits>,
    public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename bto_traits<T>::bti_traits bti_traits;

private:
    gen_bto_mult<N, bto_traits<T>, bto_mult<N, T> > m_gbto;

public:
    //! \name Constructors / destructor
    //@{

    /** \brief Constructor
        \param bta First argument
        \param btb Second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    bto_mult(
        block_tensor_rd_i<N, T> &bta,
        block_tensor_rd_i<N, T> &btb,
        bool recip = false, T c = 1.0) :

        m_gbto(bta, tensor_transf<N, T>(),
                btb, tensor_transf<N, T>(),
                recip, scalar_transf<T>(c)) {

    }

    /** \brief Constructor
        \param bta First argument
        \param pa Permutation of first argument
        \param btb Second argument
        \param pb Permutation of second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    bto_mult(
        block_tensor_rd_i<N, T> &bta, const permutation<N> &pa,
        block_tensor_rd_i<N, T> &btb, const permutation<N> &pb,
        bool recip = false, T c = 1.0) :

        m_gbto(bta, tensor_transf<N, T>(pa),
                btb, tensor_transf<N, T>(pb),
                recip, scalar_transf<T>(c)) {

    }

    /** \brief Constructor
        \param bta First argument
        \param pa Permutation of first argument
        \param btb Second argument
        \param pb Permutation of second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    bto_mult(
        block_tensor_rd_i<N, T> &bta,
        const tensor_transf<N, T> &tra,
        block_tensor_rd_i<N, T> &btb,
        const tensor_transf<N, T> &trb,
        bool recip = false,
        scalar_transf<T> trc = scalar_transf<T>()) :

        m_gbto(bta, tra, btb, trb, recip, trc) {

    }


    /** \brief Virtual destructor
     **/
    virtual ~bto_mult() { }

    //@}

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N, T> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, T> &get_schedule() const {
        return m_gbto.get_schedule();
    }

    virtual void perform(gen_block_stream_i<N, bti_traits> &out) {
        m_gbto.perform(out);
    }

    //@}

    //! \name Implementation of libtensor::additive_gen_bto<N, bti_traits>
    //@{

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc);

    virtual void perform(gen_block_tensor_i<N, bti_traits> &btc,
            const scalar_transf<T> &d);

    virtual void compute_block(
            bool zero,
            const index<N> &ic,
            const tensor_transf<N, T> &trc,
            dense_tensor_wr_i<N, T> &blkc);

    virtual void compute_block(
            const index<N> &ic,
            dense_tensor_wr_i<N, T> &blkc) {

        compute_block(true, ic, tensor_transf<N, T>(), blkc);
    }

    //@}

    virtual void perform(block_tensor_i<N, T> &btc, T d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_MULT_H
