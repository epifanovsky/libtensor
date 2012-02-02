#ifndef LIBTENSOR_LABELED_ANYTENSOR_H
#define	LIBTENSOR_LABELED_ANYTENSOR_H

#include "letter_expr.h"
#include "unassigned_expression.h"

namespace libtensor {


template<size_t N, typename T> class anytensor;


/** \brief Tensor placeholder with an attached index label
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \sa assignment_operator

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class labeled_anytensor {
private:
    anytensor<N, T> &m_t; //!< Tensor
    letter_expr<N> m_label; //!< Index label

public:
    /** \brief Constructs by attaching a label to anytensor
     **/
    labeled_anytensor(anytensor<N, T> &t, const letter_expr<N> &label) :
        m_t(t), m_label(label) {

        for(size_t i = 0; i < N; i++) for(size_t j = i + 1; j < N; j++) {
            if(m_label.letter_at(i) == m_label.letter_at(j)) {
                throw 1;
            }
        }
    }

    /** \brief Returns the tensor
     **/
    anytensor<N, T> &get_tensor() {
        return m_t;
    }

    /** \brief Returns the label
     **/
    const letter_expr<N> &get_label() const {
        return m_label;
    }

    /** \brief Performs the assignment of a tensor expression
     **/
    labeled_anytensor<N, T> &operator=(unassigned_expression<N, T> rhs);

    /** \brief Performs the assignment of a tensor expression that is
            another labeled_anytensor
     **/
    labeled_anytensor<N, T> &operator=(labeled_anytensor<N, T> rhs);

};


} // namespace libtensor

#endif // LIBTENSOR_LABELED_ANYTENSOR_H
