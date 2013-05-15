#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Element-wise multiplication operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Recip If true do element-wise division instead of multiplication.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Recip>
class mult_core : public expr_core_i<N, T> {
public:
    //! Evaluating container type
    typedef mult_eval<N, T, Recip> eval_container_t;

private:
    expr<N, T> m_expr1; //!< Left expression
    expr<N, T> m_expr2; //!< Right expression

public:
    //!    \name Construction
    //@{

    /** \brief Initializes the core with left and right expressions
     **/
    mult_core(const expr<N, T> &expr1, const expr<N, T> &expr2) :
        m_expr1(expr1), m_expr2(expr2)
    { }

    //@}

    /** \brief Returns the first expression
     **/
    expr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (const version)
     **/
    const expr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression
     **/
    expr<N, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (const version)
     **/
    const expr<N, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr1.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr1.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr1.letter_at(i);
    }

};


/** \brief Evaluates the multiplication expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Recip>
class mult_eval : public eval_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //!    Addition expression core type
    typedef mult_core<N, T, E1, E2, Recip> core_t;

    //!    Addition expression type
    typedef expr<N, T, core_t> expression_t;

    //!    Evaluating container type of the first expression (A)
    typedef typename E1::eval_container_t eval_container_a_t;

    //!    Evaluating container type of the second expression (B)
    typedef typename E2::eval_container_t eval_container_b_t;

    //!    Number of %tensor arguments in expression A
    static const size_t k_narg_tensor_a =
        eval_container_a_t::template narg<tensor_tag>::k_narg;

    //!    Number of operation arguments in expression A
    static const size_t k_narg_oper_a =
        eval_container_a_t::template narg<oper_tag>::k_narg;

    //!    Number of %tensor arguments in expression B
    static const size_t k_narg_tensor_b =
        eval_container_b_t::template narg<tensor_tag>::k_narg;

    //!    Number of operation arguments in expression B
    static const size_t k_narg_oper_b =
        eval_container_b_t::template narg<oper_tag>::k_narg;

    //! Evaluating functor type
    typedef mult_eval_functor<N, T, E1, E2, Recip,
            k_narg_tensor_a, k_narg_oper_a, k_narg_tensor_b, k_narg_oper_b>
        functor_t;

    //!    Number of arguments in the expression
    template<typename Tag, int Dummy = 0>
    struct narg {
        static const size_t k_narg = 0;
    };

private:
    functor_t m_func; //!< Sub-expression evaluation functor

public:
    mult_eval(expression_t &expr, const letter_expr<N> &label)
        throw(exception);

    virtual ~mult_eval() { }

    //!    \name Evaluation
    //@{

    void prepare();

    void clean();

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

    arg<N, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const
        throw(exception);

    //@}
};


template<size_t N, typename T, typename E1, typename E2, bool Recip>
const char *mult_eval<N, T, E1, E2, Recip>::k_clazz =
        "mult_eval<N, T, E1, E2, Recip>";

template<size_t N, typename T, typename E1, typename E2, bool Recip>
template<int Dummy>
struct mult_eval<N, T, E1, E2, Recip>::narg<oper_tag, Dummy> {
    static const size_t k_narg = 1;
};

template<size_t N, typename T, typename E1, typename E2, bool Recip>
inline mult_eval<N, T, E1, E2, Recip>::mult_eval(
    expression_t &expr, const letter_expr<N> &label)
    throw(exception) :

    m_func(expr, label) {

}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
void mult_eval<N, T, E1, E2, Recip>::prepare() {

    m_func.evaluate();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
void mult_eval<N, T, E1, E2, Recip>::clean() {

    m_func.clean();
}


template<size_t N, typename T, typename E1, typename E2, bool Recip>
template<typename Tag>
arg<N, T, Tag> mult_eval<N, T, E1, E2, Recip>::get_arg(
    const Tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const Tag&, size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}

template<size_t N, typename T, typename E1, typename E2, bool Recip>
arg<N, T, oper_tag> mult_eval<N, T, E1, E2, Recip>::get_arg(
    const oper_tag &tag, size_t i) const throw(exception) {

    static const char *method = "get_arg(const oper_tag&, size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_CORE_H
