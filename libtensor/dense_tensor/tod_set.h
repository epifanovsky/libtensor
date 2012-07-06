#ifndef LIBTENSOR_TOD_SET_H
#define LIBTENSOR_TOD_SET_H

#include <libtensor/timings.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Sets all elements of a tensor to a given value
    \tparam N Tensor order.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_set : public timings< tod_set<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_v; //!< Value

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    tod_set(double v = 0.0) : m_v(v) { }

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(dense_tensor_wr_i<N, double> &ta);

private:
    /** \brief Private copy constructor
     **/
    tod_set(const tod_set&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SET_H
