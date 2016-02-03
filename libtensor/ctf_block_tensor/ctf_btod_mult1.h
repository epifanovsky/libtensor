#ifndef LIBTENSOR_CTF_BTOD_MULT1_H
#define LIBTENSOR_CTF_BTOD_MULT1_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_mult1.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Element-wise multiplication (and division) of two distributed CTF
        block tensors
    \tparam N Tensor order.

    \sa gen_bto_mult1, btod_mult1

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_mult1 : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_mult1< N, ctf_btod_traits, ctf_btod_mult1<N> > m_gbto;

public:
    /** \brief Inititalize operation
        \param btb Block tensor B.
        \param trb Transformation of B.
        \param recip If true, perform element-wise division.
        \param c Scaling coefficient.
     **/
    ctf_btod_mult1(
        ctf_block_tensor_rd_i<N, double> &btb,
        const tensor_transf<N, double> &trb,
        bool recip = false,
        const scalar_transf<double> &c = scalar_transf<double>()) :

        m_gbto(btb, trb, recip, c) {

    }


    /** \brief Inititalize operation
        \param btb Block tensor B.
        \param recip If true, perform element-wise division.
        \param c Scaling coefficient.
     **/
    ctf_btod_mult1(
        ctf_block_tensor_rd_i<N, double> &btb,
        bool recip = false,
        double c = 1.0) :

        m_gbto(btb, tensor_transf<N, double>(), recip,
            scalar_transf<double>(c)) {

    }

    /** \brief Inititalize operation
        \param btb Block tensor B
        \param pb Permutation of B
        \param recip If true, perform element-wise division
        \param c Scaling coefficient
     **/
    ctf_btod_mult1(
        ctf_block_tensor_rd_i<N, double> &btb,
        const permutation<N> &pb,
        bool recip = false,
        double c = 1.0) :

        m_gbto(btb, tensor_transf<N, double>(pb), recip,
            scalar_transf<double>(c)) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_mult1() { }

    /** \brief Computes or adds the result to an output tensor
     **/
    virtual void perform(bool zero, gen_block_tensor_i<N, bti_traits> &btb) {

        m_gbto.perform(zero, btb);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_MULT1_H
