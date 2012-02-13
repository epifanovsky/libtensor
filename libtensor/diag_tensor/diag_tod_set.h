#ifndef LIBTENSOR_DIAG_TOD_SET_H
#define LIBTENSOR_DIAG_TOD_SET_H

#include <libtensor/timings.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Assigns all tensor entries to a value
    \param N Tensor order.

    This operation assigns all allowed tensor entries to the specified value.
    Only the elements that are allowed by the constraints of the output tensor
    are affected.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_set : public timings< diag_tod_set<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_d; //!< Assigned value

public:
    /** \brief Initializes the operation
        \param d Value to be assigned (default 0.0).
     **/
    diag_tod_set(double d = 0.0) : m_d(d) { }

    /** \brief Performs the operation
        \param ta Output tensor.
     **/
    void perform(diag_tensor_wr_i<N, double> &ta);

private:
    diag_tod_set(const diag_tod_set&);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_SET_H

