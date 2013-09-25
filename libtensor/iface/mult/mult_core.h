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
    static const char k_clazz[]; //!< Class name
private:
    expr_rhs<N, T> m_expr1; //!< Left expression
    expr_rhs<N, T> m_expr2; //!< Right expression

public:
    /** \brief Initializes the core with left and right expressions
     **/
    mult_core(const expr_rhs<N, T> &expr1, const expr_rhs<N, T> &expr2);

    /** \brief Virtual destructor
     **/
    virtual ~mult_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new mult_core<N, T, Recip>(*this);
    }

    /** \brief Returns the first expression
     **/
    expr_rhs<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (const version)
     **/
    const expr_rhs<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression
     **/
    expr_rhs<N, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (const version)
     **/
    const expr_rhs<N, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr1.get_core().contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr1.get_core().index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr1.get_core().letter_at(i);
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#include "mult_eval_functor.h"

namespace libtensor {
namespace labeled_btensor_expr {

/** \brief Evaluates the multiplication expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Recip>
class mult_eval : public eval_container_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    mult_core<N, T, Recip> m_core; //!< Expression core
    mult_eval_functor<N, T, Recip> m_func; //!< Evaluation functor

public:
    mult_eval(
        const mult_core<N, T, Recip> &core,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~mult_eval() { }

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    /** \brief Returns the number of tensors in expression
     **/
    virtual size_t get_ntensor() const {
        return 0;
    }

    /** \brief Returns the number of tensor operations in expression
     **/
    virtual size_t get_noper() const {
        return 1;
    }


    /** \brief Returns tensor arguments (not valid)
        \param i Argument number.
     **/
    virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

    /** \brief Returns operation arguments
        \param i Argument number (0 is the only valid value).
     **/
    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);
};


template<size_t N, typename T, bool Recip>
const char mult_core<N, T, Recip>::k_clazz[] = "mult_core<N, T, Recip>";


template<size_t N, typename T, bool Recip>
mult_core<N, T, Recip>::mult_core(
    const expr_rhs<N, T> &expr1, const expr_rhs<N, T> &expr2) :

    m_expr1(expr1), m_expr2(expr2) {

    static const char method[] =
        "mult_core(const expr<N, T>&, const expr<N, T>&)";

    for(size_t i = 0; i < N; i++) {
        const letter &l = expr1.get_core().letter_at(i);
        if(! expr2.get_core().contains(l)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Unique index in A.");
        }
    }
}


template<size_t N, typename T, bool Recip>
eval_container_i<N, T> *mult_core<N, T, Recip>::create_container(
    const letter_expr<N> &label) const {

    return new mult_eval<N, T, Recip>(*this, label);
}


template<size_t N, typename T, bool Recip>
const char mult_eval<N, T, Recip>::k_clazz[] = "mult_eval<N, T, Recip>";


template<size_t N, typename T, bool Recip>
mult_eval<N, T, Recip>::mult_eval(
    const mult_core<N, T, Recip> &core,
    const letter_expr<N> &label) :

    m_core(core), m_func(m_core, label) {

}


template<size_t N, typename T, bool Recip>
void mult_eval<N, T, Recip>::prepare() {

    m_func.evaluate();
}


template<size_t N, typename T, bool Recip>
void mult_eval<N, T, Recip>::clean() {

    m_func.clean();
}


template<size_t N, typename T, bool Recip>
arg<N, T, tensor_tag> mult_eval<N, T, Recip>::get_tensor_arg(size_t i) {

    static const char *method = "get_tensor_arg(size_t)";
    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}

template<size_t N, typename T, bool Recip>
arg<N, T, oper_tag> mult_eval<N, T, Recip>::get_oper_arg(size_t i) {

    static const char *method = "get_oper_arg(size_t)";

    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    return m_func.get_arg();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_MULT_CORE_H
