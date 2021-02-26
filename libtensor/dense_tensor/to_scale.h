#ifndef LIBTENSOR_TO_SCALE_H
#define LIBTENSOR_TO_SCALE_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Scales tensor elements by a constant
    \tparam N Tensor order.

    \ingroup libtensor_dense_tensor_to
 **/
template<size_t N, typename T>
class to_scale: public timings< to_scale<N, T> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    T m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param c Scaling coefficient.
     **/
    to_scale(const scalar_transf<T> &c);

    /** \brief Initializes the operation
        \param c Scaling coefficient.
     **/
    to_scale(T c);

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(dense_tensor_wr_i<N, T> &ta);
};


} // namespace libtensor

#endif // LIBTENSOR_TO_SCALE_H
