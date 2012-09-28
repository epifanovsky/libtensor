#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_mult.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult : public additive_bto<N, btod_traits>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

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

    virtual void perform(block_tensor_i<N, double> &btc);
    virtual void perform(block_tensor_i<N, double> &btc, const double &d);

    virtual void compute_block(dense_tensor_i<N, double> &blkc,
        const index<N> &ic);

    virtual void compute_block(bool zero,
        dense_tensor_i<N, double> &blkc, const index<N> &ic,
        const tensor_transf<N, double> &trc, const double &c);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
