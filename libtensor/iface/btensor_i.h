#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include "../defs.h"
#include "../exception.h"
#include <libtensor/block_tensor/block_tensor_i.h>
#include "labeled_btensor.h"

namespace libtensor {


/** \brief Block tensor interface
    \tparam N Block %tensor order.
    \tparam T Block %tensor element type.

    \ingroup libtensor_iface
**/
template<size_t N, typename T>
class btensor_i : virtual public block_tensor_i<N, T> {

    /** \brief Attaches a label to this %tensor and returns it as a
                labeled %tensor
     **/
    labeled_btensor<N, T, false> operator()(letter_expr<N> expr);
};


/** \brief Specialization for N = 1

    \ingroup libtensor_iface
**/
template<typename T>
class btensor_i<1, T> : virtual public block_tensor_i<1, T> {

    labeled_btensor<1, T, false> operator()(letter_expr<1> expr);

    labeled_btensor<1, T, false> operator()(const letter &expr);
};


template<size_t N, typename T>
inline labeled_btensor<N, T, false> btensor_i<N, T>::operator()(
    letter_expr<N> expr) {

    return labeled_btensor<N, T, false>(*this, expr);
}


template<typename T>
inline labeled_btensor<1, T, false> btensor_i<1, T>::operator()(
    letter_expr<1> expr) {

    return labeled_btensor<1, T, false>(*this, expr);
}


template<typename T>
inline labeled_btensor<1, T, false> btensor_i<1, T>::operator()(
    const letter &let) {

    return labeled_btensor<1, T, false>(*this, letter_expr<1>(let));
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H

