#ifndef LIBTENSOR_BTOD_MULT_H
#define LIBTENSOR_BTOD_MULT_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult :
    public additive_bto<N, btod_traits>,
    public timings< btod_mult<N> > {

public:
    static const char *k_clazz; //!< Class name

private:
    block_tensor_i<N, double> &m_bta; //!< First argument
    block_tensor_i<N, double> &m_btb; //!< Second argument
    permutation<N> m_pa; //!< Permutation of bta
    permutation<N> m_pb; //!< Permutation of btb
    bool m_recip; //!< Reciprocal
    double m_c; //!< Scaling coefficient

    block_index_space<N> m_bis; //!< Block %index space of the result
    symmetry<N, double> m_sym; //!< Result symmetry
    assignment_schedule<N, double> m_sch; //!< Schedule

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
    btod_mult(block_tensor_i<N, double> &bta, block_tensor_i<N, double> &btb,
            bool recip = false, double c = 1.0);

    /** \brief Constructor
        \param bta First argument
        \param pa Permutation of first argument
        \param btb Second argument
        \param pb Permutation of second argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    btod_mult(block_tensor_i<N, double> &bta, const permutation<N> &pa,
            block_tensor_i<N, double> &btb, const permutation<N> &pb,
            bool recip = false, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_mult();

    //@}



    //!    \name Implementation of
    //      libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bta.get_bis();
    }

    virtual const symmetry<N, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<N, double> &get_schedule() const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    //@}

    virtual void perform(bto_stream_i<N, btod_traits> &out);
    virtual void perform(block_tensor_i<N, double> &btc);
    virtual void perform(block_tensor_i<N, double> &btc, const double &d);

    virtual void compute_block(bool zero, dense_tensor_i<N, double> &blk,
        const index<N> &idx, const tensor_transf<N, double> &tr,
        const double &c);

private:
    btod_mult(const btod_mult<N> &);
    const btod_mult<N> &operator=(const btod_mult<N> &);

    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT_H
