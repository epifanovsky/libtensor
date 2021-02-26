#ifndef LIBTENSOR_BTO_MULT1_H
#define LIBTENSOR_BTO_MULT1_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_mult1.h>

namespace libtensor {


/** \brief Elementwise multiplication of two block tensors
    \tparam N Tensor order.

    \sa tod_mult1

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_mult1 : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_mult1< N, bto_traits<T>, bto_mult1<N, T> > m_gbto;

public:
    /** \brief Inititalize operation
        \param btb Block tensor B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    bto_mult1(block_tensor_rd_i<N, T> &btb,
            const tensor_transf<N, T> &trb, bool recip = false,
            const scalar_transf<T> &c = scalar_transf<T>());


    /** \brief Inititalize operation
        \param btb Block tensor B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    bto_mult1(block_tensor_rd_i<N, T> &btb,
            bool recip = false, T c = 1.0);

    /** \brief Inititalize operation
        \param btb Block tensor B
        \param pb Permutation of B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    bto_mult1(block_tensor_rd_i<N, T> &btb, const permutation<N> &pb,
            bool recip = false, T c = 1.0);

    /** \brief Perform the operation
        \param zero If true, perform \f$ A = A * B \f$ otherwise \f$ A = A + A * B \f$
        \param bta Result tensor A
     **/
    void perform(bool zero, block_tensor_i<N, T> &bta);
};


} // namespace libtensor

#endif // LIBTENSOR_BTO_MULT1_H
