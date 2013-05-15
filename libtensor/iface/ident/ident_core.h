#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Identity expression core (references one labeled tensor)
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the tensor is an l-value.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class ident_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //! Labeled block tensor type
    typedef labeled_btensor<N, T, Assignable> labeled_btensor_t;

    //! Evaluating container type
    typedef ident_eval<N, T, Assignable> eval_container_t;

private:
    labeled_btensor_t m_t; //!< Labeled block tensor

public:
    /** \brief Initializes the operation with a tensor reference
     **/
    ident_core(const labeled_btensor_t &t) : m_t(t) { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new ident_core(*this);
    }

    /** \brief Returns the labeled block tensor
     **/
    labeled_btensor_t &get_tensor() {
        return m_t;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

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
const char ident_core<N, T, Assignable>::k_clazz[] =
    "ident_core<N, T, Assignable>";


/** \brief Evaluating container for a labeled tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Assignable Whether the tensor is an l-value.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable>
class ident_eval : public eval_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //!    Expression core type
    typedef core_ident<N, T, Assignable> core_t;

    //!    Expression type
    typedef expr<N, T, core_t> expression_t;

    //!    Number of arguments
    template<typename Tag, int Dummy = 0>
    struct narg {
        static const size_t k_narg = 0;
    };

private:
    expr<N, T> m_expr;
    ident_core<N, T, A> &m_core;
    permutation<N> m_perm;

public:
    ident_eval(expr<N, T> &e, const letter_expr<N> &label);

    virtual ~ident_eval();

    //!    \name Evaluation
    //@{

    void prepare() { }

    void clean() { }

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

    /** \brief Returns the %tensor argument
        \param i Argument number (0 is the only allowed value)
     **/
    arg<N, T, tensor_tag> get_arg(const tensor_tag &tag, size_t i) const
        throw(exception);

    //@}
};


template<size_t N, typename T, bool A>
const char ident_eval<N, T, A>::k_clazz[] = "ident_eval<N, T, A>";


template<size_t N, typename T, bool Assignable>
template<int Dummy>
struct ident_eval<N, T, Assignable>::narg<tensor_tag, Dummy> {
    static const size_t k_narg = 1;
};


template<size_t N, typename T, bool A>
ident_eval<N, T, A>::ident_eval(
    expr<N, T> &e, const letter_expr<N> &label) :

    m_expr(e),
    m_core(dynamic_cast< ident_core<N, T, A>& >(m_expr.get_core())),
    m_perm(label.permutation_of(m_core.get_tensor().get_label())) {

}


template<size_t N, typename T, bool Assignable>
template<typename Tag>
inline arg<N, T, Tag> ident_eval<N, T, Assignable>::get_arg(
    const Tag &tag, size_t i) const throw(exception) {

    static const char method[] = "get_arg(const Tag&, size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, typename T, bool Assignable>
inline arg<N, T, tensor_tag> ident_eval<N, T, Assignable>::get_arg(
    const tensor_tag &tag, size_t i) const throw(exception) {

    static const char method[] = "get_arg(const tensor_tag&, size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }
    return arg<N, T, tensor_tag>(
        m_expr.get_core().get_tensor().get_btensor(), m_perm, 1.0);
}


template<size_t N, typename T, bool A>
eval_container_i<N, T> *ident_core<N, T, A>::create_container(
    const letter_expr<N> &label) const {

    return new ident_eval<N, T, A>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_IDENT_CORE_H
