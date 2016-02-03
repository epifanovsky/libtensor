#ifndef LIBTENSOR_CTF_TOD_MULT_H
#define LIBTENSOR_CTF_TOD_MULT_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Element-wise multiplication or division of a CTF distributed tensor
    \tparam N Tensor order.

    \sa tod_mult

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_mult : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_dense_tensor_i<N, double> &m_ta; //!< First argument (A)
    ctf_dense_tensor_i<N, double> &m_tb; //!< Second argument (B)
    tensor_transf<N, double> m_tra; //!< Transformation of A
    tensor_transf<N, double> m_trb; //!< Transformation of B
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    double m_c; //!< Scaling coefficient
    dimensions<N> m_dimsc; //!< Result dimensions

public:
    /** \brief Initializes the operation
        \param ta First argument.
        \param tra Tensor transformation of ta with respect to result.
        \param tb Second argument.
        \param trb Tensor transformation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation
     **/
    ctf_tod_mult(
        ctf_dense_tensor_i<N, double> &ta,
        const tensor_transf<N, double> &tra,
        ctf_dense_tensor_i<N, double> &tb,
        const tensor_transf<N, double> &trb,
        bool recip,
        const scalar_transf<double> &trc = scalar_transf<double>());

    /** \brief Creates the operation
        \param ta First argument.
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param coeff Scaling coefficient
     **/
    ctf_tod_mult(
        ctf_dense_tensor_i<N, double> &ta,
        ctf_dense_tensor_i<N, double> &tb,
        bool recip = false,
        double c = 1.0);

    /** \brief Initializes the operation
        \param ta First argument.
        \param pa Permutation of ta with respect to result.
        \param tb Second argument.
        \param pb Permutation of tb with respect to result.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param trc Scalar transformation
     **/
    ctf_tod_mult(
        ctf_dense_tensor_i<N, double> &ta,
        const permutation<N> &pa,
        ctf_dense_tensor_i<N, double> &tb,
        const permutation<N> &pb,
        bool recip, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_mult() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param tc Output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_MULT_H
