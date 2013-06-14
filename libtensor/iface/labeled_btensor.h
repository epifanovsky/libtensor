#ifndef LIBTENSOR_LABELED_BTENSOR_H
#define    LIBTENSOR_LABELED_BTENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "labeled_btensor_base.h"

namespace libtensor {

namespace labeled_btensor_expr {
template<size_t N, typename T> class expr;
} // namespace labeled_btensor_expr

template<size_t N, typename T> class btensor_i;


/** \brief Block %tensor with an attached label
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the %tensor can be an l-value.

    \ingroup libtensor_iface
 **/
template<size_t N, typename T, bool Assignable> class labeled_btensor;


template<size_t N, typename T>
class labeled_btensor<N, T, false> : public labeled_btensor_base<N> {
private:
    btensor_rd_i<N, T> &m_bt;

public:
    labeled_btensor(btensor_rd_i<N, T> &bt, const letter_expr<N> &label) :
        labeled_btensor_base<N>(label), m_bt(bt) { }

    btensor_rd_i<N, T> &get_btensor() { return m_bt; }
};

/** \brief Partial specialization of the assignable labeled tensor

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_btensor<N, T, true> : public labeled_btensor_base<N> {
private:
    btensor_i<N, T> &m_bt;
public:
    labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N> &label) :
        labeled_btensor_base<N>(label), m_bt(bt) { }

    btensor_i<N, T> &get_btensor() { return m_bt; }

    /** \brief Assigns this %tensor to an expression
     **/
    labeled_btensor<N, T, true> &operator=(
        labeled_btensor_expr::expr<N, T> rhs);

    template<bool Assignable>
    labeled_btensor<N, T, true> &operator=(
        labeled_btensor<N, T, Assignable> rhs);


};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

