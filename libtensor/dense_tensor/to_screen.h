#ifndef LIBTENSOR_TOD_SCREEN_H
#define LIBTENSOR_TOD_SCREEN_H

#include <cmath>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Screens a tensor for a certain value of element
    \tparam N Tensor order.

    The operation goes over tensor elements and searches for a given
    element value within a threshold. If requested, the values that match
    within the threshold are replaced with the exact value.

    The return value indicates whether one or more elements matched.

    \ingroup libtensor_dense_tensor_to
 **/
template<size_t N, typename T>
class to_screen : public timings< to_screen<N, T> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    T m_a; //!< Value
    T m_thresh; //!< Equality threshold

public:
    /** \brief Initializes the operation
        \param a Element value (default 0.0).
        \param thresh Threshold (default 0.0 -- exact match).
     **/
    to_screen(T a = 0.0, T thresh = 0.0) :
        m_a(a), m_thresh(std::abs(thresh))
    { }

    /** \brief Screens the elements for the given value
        \param t Tensor.
        \return True if a match is found, false otherwise.
     **/
    bool perform_screen(dense_tensor_rd_i<N, T> &t);

    /** \brief Screens and replaces the matches with the exact value
        \param t Tensor.
        \return True if a match is found, false otherwise.
     **/
    bool perform_replace(dense_tensor_wr_i<N, T> &t);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCREEN_H
