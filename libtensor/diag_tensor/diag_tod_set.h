#ifndef LIBTENSOR_DIAG_TOD_SET_H
#define LIBTENSOR_DIAG_TOD_SET_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "diag_tensor_i.h"

namespace libtensor {


/** \brief Changes all tensor entries by or to a value
    \param N Tensor order.

    This operation changes all allowed tensor entries by or to the specified
    value. Only the elements that are allowed by the constraints of the output
    tensor are affected.

    \ingroup libtensor_diag_tensor
 **/
template<size_t N>
class diag_tod_set : public timings< diag_tod_set<N> >, public noncopyable {
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
        \param zero Zero tensor first
        \param ta Output tensor.
     **/
    void perform(bool zero, diag_tensor_wr_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_TOD_SET_H

