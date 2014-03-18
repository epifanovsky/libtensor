#ifndef LIBTENSOR_CTF_TOD_MULT1_H
#define LIBTENSOR_CTF_TOD_MULT1_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Element-wise multiplication or division of a distributed CTF tensor
    \tparam N Tensor order.

    \sa tod_mult1

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_mult1 : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_dense_tensor_i<N, double> &m_tb; //!< Second argument (B)
    permutation<N> m_permb; //!< Permutation of B
    bool m_recip; //!< Reciprocal (multiplication by 1/bi)
    double m_c; //!< Scaling coefficient

public:
    /** \brief Creates the operation
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Scalar transformation
     **/
    ctf_tod_mult1(
        ctf_dense_tensor_i<N, double> &tb,
        const tensor_transf<N, double> &trb, bool recip = false,
        const scalar_transf<double> &c = scalar_transf<double>());

    /** \brief Creates the operation
        \param tb Second argument.
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    ctf_tod_mult1(
        ctf_dense_tensor_i<N, double> &tb,
        bool recip = false, double c = 1.0);

    /** \brief Creates the operation
        \param tb Second argument.
        \param p Permutation of argument
        \param recip \c false (default) sets up multiplication and
            \c true sets up element-wise division.
        \param c Coefficient
     **/
    ctf_tod_mult1(
        ctf_dense_tensor_i<N, double> &tb, const permutation<N> &p,
        bool recip = false, double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_mult1() { }

    /** \brief Runs the operation
        \param zero Overwrite/add to flag.
        \param ta Input/output tensor.
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_MULT1_H
