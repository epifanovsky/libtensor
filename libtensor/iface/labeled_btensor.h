#ifndef LIBTENSOR_LABELED_BTENSOR_H
#define    LIBTENSOR_LABELED_BTENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "labeled_btensor_base.h"

namespace libtensor {

namespace labeled_btensor_expr {
template<size_t N, typename T> class expr;
} // namespace labeled_btensor_expr


/** \brief Block %tensor with an attached label
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the %tensor can be an l-value.

    \ingroup libtensor_iface
 **/
template<size_t N, typename T, bool Assignable>
class labeled_btensor : public labeled_btensor_rd_base<N, T> {
public:
    labeled_btensor(btensor_rd_i<N, T> &bt, const letter_expr<N> &label) :
        labeled_btensor_rd_base<N, T>(bt, label) { }
};

/** \brief Partial specialization of the assignable labeled tensor

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_btensor<N, T, true> : public labeled_btensor_rd_base<N, T> {
private:
    btensor_i<N, T> &m_bt;
public:
    labeled_btensor(btensor_i<N, T> &bt, const letter_expr<N> &label) :
        labeled_btensor_rd_base<N, T>(bt, label), m_bt(bt) { }

    btensor_i<N, T> &get_btensor() { return m_bt; }

    /** \brief Assigns this %tensor to an expression
     **/
    labeled_btensor<N, T, true> &operator=(
        const labeled_btensor_expr::expr<N, T> rhs);

    template<bool AssignableR>
    labeled_btensor<N, T, true> &operator=(
        labeled_btensor<N, T, AssignableR> rhs);

    labeled_btensor<N, T, true> &operator=(
        labeled_btensor<N, T, true> rhs);

};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_H

