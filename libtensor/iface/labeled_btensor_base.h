#ifndef LIBTENSOR_LABELED_BTENSOR_BASE_H
#define LIBTENSOR_LABELED_BTENSOR_BASE_H

#include "../defs.h"
#include "../exception.h"
#include "letter.h"
#include "letter_expr.h"

namespace libtensor {

template<size_t N, typename T> class btensor_rd_i;
template<size_t N, typename T> class btensor_i;

/** \brief Block %tensor with an attached label (base class)
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Label Label expression.

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class labeled_btensor_rd_base {
private:
    typedef T element_t;
    typedef letter_expr<N> label_t;

private:
    btensor_rd_i<N, T> &m_bt;
    letter_expr<N> m_label;

public:
    /** \brief Constructs the labeled block %tensor
     **/
    labeled_btensor_rd_base(btensor_rd_i<N, T> &bt,
            const letter_expr<N> &label) : m_bt(bt), m_label(label) {

    }

    /** \brief Returns the tensor interface
     **/
    btensor_rd_i<N, T> &get_btensor() const {
        return m_bt;
    }

    /** \brief Returns the label
     **/
    const letter_expr<N> &get_label() const {
        return m_label;
    }

    /** \brief Returns whether the label contains a %letter
     **/
    bool contains(const letter &let) const {
        return m_label.contains(let);
    }

    /** \brief Returns the %index of a %letter in the label
     **/
    size_t index_of(const letter &let) const throw(exception) {
        return m_label.index_of(let);
    }

    /** \brief Returns the %letter at a given position in the label
     **/
    const letter &letter_at(size_t i) const throw(exception) {
        return m_label.letter_at(i);
    }

};

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_BASE_H
