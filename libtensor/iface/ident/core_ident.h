#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_IDENT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_IDENT_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, bool Assignable> class eval_ident;


/** \brief Identity expression core (references one labeled tensor)
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the tensor is an l-value.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class core_ident : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Labeled block tensor type
    typedef labeled_btensor<N, T, Assignable> labeled_btensor_t;

    //! Evaluating container type
    typedef eval_ident<N, T, Assignable> eval_container_t;

private:
    labeled_btensor_t m_t; //!< Labeled block tensor

public:
    /** \brief Initializes the operation with a tensor reference
     **/
    core_ident(const labeled_btensor_t &t) : m_t(t) { }

    /** \brief Returns the labeled block tensor
     **/
    labeled_btensor_t &get_tensor() {
        return m_t;
    }

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_t.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_t.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_t.letter_at(i);
    }

};


template<size_t N, typename T, bool Assignable>
const char core_ident<N, T, Assignable>::k_clazz[] =
    "core_ident<N, T, Assignable>";


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_IDENT_H
