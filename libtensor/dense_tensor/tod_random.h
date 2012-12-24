#ifndef LIBTENSOR_TOD_RANDOM_H
#define LIBTENSOR_TOD_RANDOM_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Fills a tensor with random numbers or adds them to it
    \tparam N Tensor order.

    This operation either fills a tensor with random numbers equally
    distributed in the intervall [0;1[ or adds those numbers to the tensor
    scaled by a coefficient.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_random : public timings< tod_random<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_c; // Scaling coefficient

public:
    /** \brief Prepares the operation
        \param Scalar transformation
     **/
    tod_random(const scalar_transf<double> &c = scalar_transf<double>());

    /** \brief Prepares the operation
        \param Scaling coefficient
     **/
    tod_random(double c);

    /** \brief Perform operation
        \param zero Zero tensor first
        \param t Tensor to put random data
     **/
    void perform(bool zero, dense_tensor_wr_i<N, double> &t);

    /** \brief Perform operation
        \param t Tensor to put random data
     **/
    void perform(dense_tensor_wr_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_H
