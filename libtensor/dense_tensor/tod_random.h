#ifndef LIBTENSOR_TOD_RANDOM_H
#define LIBTENSOR_TOD_RANDOM_H

#include <libtensor/timings.h>
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
class tod_random : public timings< tod_random<N> > {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Prepares the operation
     **/
    tod_random();

    void perform(bool zero, double c, dense_tensor_wr_i<N, double> &t);
    void perform(dense_tensor_wr_i<N, double> &t);
    void perform(dense_tensor_wr_i<N, double> &t, double c);

private:
    static void update_seed(); //! updates the seed value by using srand48

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_RANDOM_H
