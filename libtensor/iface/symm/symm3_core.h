#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression core for the symmetrization over three indexes
    \tparam N Tensor order.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T>
class symm3_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    const letter &m_l1; //!< First %index
    const letter &m_l2; //!< Second %index
    const letter &m_l3; //!< Third %index
    expr<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param l1 First symmetrized %index.
        \param l2 Second symmetrized %index.
        \param l3 Third symmetrized %index.
        \param subexpr Sub-expression.
     **/
    symm3_core(const letter &l1, const letter &l2, const letter &l3,
        const expr<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~symm3_core() { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new symm3_core(*this);
    }

    /** \brief Returns the first symmetrized index
     **/
    const letter &get_l1() const {
        return m_l1;
    }

    /** \brief Returns the second symmetrized index
     **/
    const letter &get_l2() const {
        return m_l2;
    }

    /** \brief Returns the third symmetrized index
     **/
    const letter &get_l3() const {
        return m_l3;
    }

    /** \brief Returns the sub-expression
     **/
    expr<N, T> &get_sub_expr() {
        return m_expr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr<N, T> &get_sub_expr() const {
        return m_expr;
    }

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    bool contains(const letter &let) const {
        return m_expr.contains(let);
    }

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    size_t index_of(const letter &let) const {
        return m_expr.index_of(let);
    }

    /** \brief Returns the letter at a given position in the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    const letter &letter_at(size_t i) const {
        return m_expr.letter_at(i);
    }

};


/** \brief Evaluating container for the symmetrization three indexes
    \tparam N Tensor order.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T>
class symm3_eval : public eval_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

    //!    Expression core type
    typedef symm3_core<N, Sym, T, SubCore> core_t;

    //!    Expression type
    typedef expr<N, T, core_t> expression_t;

    //!    Sub-expression core type
    typedef SubCore sub_core_t;

    //!    Sub-expression type
    typedef expr<N, T, sub_core_t> sub_expr_t;

    //!    Evaluating container type
    typedef typename sub_expr_t::eval_container_t sub_eval_container_t;

    //!    Number of tensor arguments
    static const size_t k_sub_narg_tensor =
        sub_eval_container_t::template narg<tensor_tag>::k_narg;

    //!    Number of operation arguments
    static const size_t k_sub_narg_oper =
        sub_eval_container_t::template narg<oper_tag>::k_narg;

    //!    Evaluation functor type
    typedef evalfunctor<N, T, sub_core_t, k_sub_narg_tensor,
        k_sub_narg_oper> sub_evalfunctor_t;

    //!    Number of arguments in the expression
    template<typename Tag, int Dummy = 0>
    struct narg {
        static const size_t k_narg = 0;
    };

private:
    sub_expr_t m_sub_expr; //!< Sub-expression
    sub_eval_container_t m_sub_eval_cont; //!< Evaluation of the sub-expression
    sub_evalfunctor_t m_sub_eval; //!< Evaluation functor
    size_t m_i1, m_i2, m_i3; //!< Symmetrization indexes
    btod_symmetrize3<N> *m_op; //!< Symmetrization operation
    arg<N, T, oper_tag> *m_arg; //!< Argument

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    symm3_eval(const expr<N, T> &e, const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~symm3_eval();

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const;

    /** \brief Returns the operation argument
     **/
    arg<N, T, oper_tag> get_arg(const oper_tag &tag, size_t i) const;

private:
    void create_arg();
    void destroy_arg();

};


template<size_t N, bool Sym, typename T>
const char symm3_core<N, Sym, T>::k_clazz[] = "symm3_core<N, Sym, T>";


template<size_t N, bool Sym, typename T>
symm3_core<N, Sym, T>::symm3_core(const letter &l1, const letter &l2,
    const letter &l3, const expr<N, T> &subexpr) :

    m_l1(l1), m_l2(l2), m_l3(l3), m_subexpr(subexpr) {

    static const char method[] = "symm3_core(const letter&, "
        "const letter&, const letter&, const expr<N, T>&)";

    if(m_l1 == m_l2 || m_l1 == m_l3 || m_l2 == m_l3) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Symmetrized indexes must be different.");
    }
}


template<size_t N, bool Sym, typename T>
const char symm3_eval<N, Sym, T>::k_clazz[] = "symm3_eval<N, Sym, T>";


template<size_t N, bool Sym, typename T, typename SubCore>
template<int Dummy>
struct symm3_eval<N, Sym, T, SubCore>::narg<oper_tag, Dummy> {
    static const size_t k_narg = 1;
};


template<size_t N, bool Sym, typename T>
symm3_eval<N, Sym, T>::symm3_eval(
    const expr<N, T> &e,
    const letter_expr<N> &label) :

    m_sub_expr(expr.get_core().get_sub_expr()),
    m_sub_eval_cont(m_sub_expr, label),
    m_sub_eval(m_sub_expr, m_sub_eval_cont),
    m_op(0), m_arg(0) {

    m_i1 = label.index_of(expr.get_core().get_l1());
    m_i2 = label.index_of(expr.get_core().get_l2());
    m_i3 = label.index_of(expr.get_core().get_l3());
}


template<size_t N, bool Sym, typename T>
symm3_eval<N, Sym, T>::~symm3_eval() {

    destroy_arg();
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T>::prepare() {

    m_sub_eval_cont.prepare();
    create_arg();
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T>::clean() {

    destroy_arg();
    m_sub_eval_cont.clean();
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T>::create_arg() {

    destroy_arg();
    m_op = new btod_symmetrize3<N>(m_sub_eval.get_bto(),
        m_i1, m_i2, m_i3, Sym);
    m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T, SubCore>::destroy_arg() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


template<size_t N, bool Sym, typename T>
template<typename Tag>
arg<N, T, Tag> symm3_eval<N, Sym, T>::get_arg(const Tag &tag,
    size_t i) const {

    static const char *method = "get_arg(const Tag&, size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, bool Sym, typename T>
arg<N, T, oper_tag> symm3_eval<N, Sym, T>::get_arg(
    const oper_tag &tag, size_t i) const {

    static const char *method = "get_arg(const oper_tag&, size_t)";
    if(i == 0) {
        return *m_arg;
    } else {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
