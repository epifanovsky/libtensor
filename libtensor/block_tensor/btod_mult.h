#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/gen_block_tensor/gen_bto_mult.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_mult :
    public additive_gen_bto<N, btod_traits::bti_traits>,
    public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename btod_traits::bti_traits bti_traits;

private:
    gen_bto_mult<N, btod_traits, btod_mult<N> > m_gbto;

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
    btod_mult(
        block_tensor_rd_i<N, double> &bta,
        block_tensor_rd_i<N, double> &btb,
        bool recip = false, double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(),
                btb, tensor_transf<N, double>(),
                recip, scalar_transf<double>(c)) {

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
    btod_mult(
        block_tensor_rd_i<N, double> &bta, const permutation<N> &pa,
        block_tensor_rd_i<N, double> &btb, const permutation<N> &pb,
        bool recip = false, double c = 1.0) :

        m_gbto(bta, tensor_transf<N, double>(pa),
                btb, tensor_transf<N, double>(pb),
                recip, scalar_transf<double>(c)) {

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
    btod_mult(
        block_tensor_rd_i<N, double> &bta,
        const tensor_transf<N, double> &tra,
        block_tensor_rd_i<N, double> &btb,
        const tensor_transf<N, double> &trb,
        bool recip = false,
        scalar_transf<double> trc = scalar_transf<double>()) :

        m_gbto(bta, tra, btb, trb, recip, trc) {

    }


    /** \brief Virtual destructor
     **/
    virtual ~btod_mult() { }

    //@}

    //! \name Implementation of libtensor::direct_gen_bto<N, bti_traits>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_gbto.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_gbto.get_symmetry();
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
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
            const scalar_transf<double> &d);

    virtual void compute_block(
            bool zero,
            const index<N> &ic,
            const tensor_transf<N, double> &trc,
            dense_tensor_wr_i<N, double> &blkc);

    virtual void compute_block(
            const index<N> &ic,
            dense_tensor_wr_i<N, double> &blkc) {

        compute_block(true, ic, tensor_transf<N, double>(), blkc);
    }

    //@}

    virtual void perform(block_tensor_i<N, double> &btc, double d);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
