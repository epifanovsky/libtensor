#ifndef LIBTENSOR_TOD_SCREEN_H
#define LIBTENSOR_TOD_SCREEN_H

#include <cmath>
#include <libtensor/timings.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Screens a tensor for a certain value of element
    \tparam N Tensor order.

    The operation goes over tensor elements and searches for a given
    element value within a threshold. If requested, the values that match
    within the threshold are replaced with the exact value.

    The return value indicates whether one or more elements matched.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_screen : public timings< tod_screen<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    double m_a; //!< Value
    double m_thresh; //!< Equality threshold

public:
    /** \brief Initializes the operation
        \param a Element value (default 0.0).
        \param thresh Threshold (default 0.0 -- exact match).
     **/
    tod_screen(double a = 0.0, double thresh = 0.0) :
        m_a(a), m_thresh(fabs(thresh))
    { }

    /** \brief Screens the elements for the given value
        \param t Tensor.
        \return True if a match is found, false otherwise.
     **/
    bool perform_screen(dense_tensor_rd_i<N, double> &t);

    /** \brief Screens and replaces the matches with the exact value
        \param t Tensor.
        \return True if a match is found, false otherwise.
     **/
    bool perform_replace(dense_tensor_wr_i<N, double> &t);

private:
    tod_screen(const tod_screen&);
    const tod_screen &operator=(const tod_screen&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCREEN_H
