#ifndef LIBTENSOR_TOD_SCALE_H
#define LIBTENSOR_TOD_SCALE_H

#include <libtensor/timings.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Scales tensor elements by a constant
    \tparam N Tensor order.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_scale: public timings< tod_scale<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param c Scaling coefficient.
     **/
    tod_scale(double c) : m_c(c) { }

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &ta);

private:
    /** \brief Private copy constructor
     **/
    tod_scale(const tod_scale&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCALE_H
