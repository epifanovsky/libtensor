#ifndef LIBTENSOR_CTF_TOD_RANDOM_H
#define LIBTENSOR_CTF_TOD_RANDOM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Fills a distributed tensor with random values
    \tparam N Tensor order.

    \sa tod_random

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_random : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    double m_c; // Scaling coefficient

public:
    /** \brief Prepares the operation
        \param Scalar transformation
     **/
    ctf_tod_random(const scalar_transf<double> &c = scalar_transf<double>());

    /** \brief Prepares the operation
        \param Scaling coefficient
     **/
    ctf_tod_random(double c);

    /** \brief Perform operation
        \param zero Zero tensor first
        \param t Tensor to put random data
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_RANDOM_H
