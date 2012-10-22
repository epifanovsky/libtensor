#ifndef LIBTENSOR_BTOD_MULT1_H
#define LIBTENSOR_BTOD_MULT1_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_mult1.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \sa tod_mult1

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_mult1 : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_mult1< N, btod_traits, btod_mult1<N> > m_gbto;

public:
    /** \brief Inititalize operation
        \param btb Block tensor B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    btod_mult1(block_tensor_i<N, double> &btb,
            const tensor_transf<N, double> &trb, bool recip = false,
            const scalar_transf<double> &c = scalar_transf<double>());


    /** \brief Inititalize operation
        \param btb Block tensor B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    btod_mult1(block_tensor_i<N, double> &btb,
            bool recip = false, double c = 1.0);

    /** \brief Inititalize operation
        \param btb Block tensor B
        \param pb Permutation of B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    btod_mult1(block_tensor_i<N, double> &btb, const permutation<N> &pb,
            bool recip = false, double c = 1.0);

    /** \brief Perform the operation
        \param zero If true, perform \f$ A = A * B \f$ otherwise \f$ A = A + A * B \f$
        \param bta Result tensor A
     **/
    void perform(bool zero, block_tensor_i<N, double> &bta);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_MULT1_H
