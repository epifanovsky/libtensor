#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include "../defs.h"
#include "../exception.h"
#include <libtensor/block_tensor/block_tensor_i.h>
#include "labeled_btensor.h"

namespace libtensor {


/** \brief Block tensor interface (read-only)
    \tparam N Block %tensor order.
    \tparam T Block %tensor element type.

    \ingroup libtensor_iface
**/
template<size_t N, typename T>
class btensor_rd_i : virtual public block_tensor_rd_i<N, T> {
public:
    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_btensor<N, T, false> operator()(letter_expr<N> expr);
};


template<typename T>
class btensor_rd_i<1, T> : virtual public block_tensor_rd_i<1, T> {
public:
    labeled_btensor<1, T, false> operator()(const letter &let);
};


/** \brief Block tensor interface
    \tparam N Block %tensor order.
    \tparam T Block %tensor element type.

    \ingroup libtensor_iface
**/
template<size_t N, typename T>
class btensor_i :
    virtual public btensor_rd_i<N, T>,
    virtual public block_tensor_i<N, T> {

};


template<size_t N, typename T>
inline labeled_btensor<N, T, false> btensor_rd_i<N, T>::operator()(
    letter_expr<N> expr) {

    return labeled_btensor<N, T, false>(*this, expr);
}


template<typename T>
inline labeled_btensor<1, T, false> btensor_rd_i<1, T>::operator()(
    const letter &let) {

    return labeled_btensor<1, T, false>(*this, letter_expr<1>(let));
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H

