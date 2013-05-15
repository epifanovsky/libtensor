#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression core for the symmetrization over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in the set.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, bool Sym, typename T>
class symm2_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
     //!    Evaluating container type
    typedef symm2_eval<N, M, Sym, T> eval_container_t;

private:
    letter_expr<M> m_sym1; //!< First set of symmetrized indexes
    letter_expr<M> m_sym2; //!< Second set of symmetrized indexes
    expr<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param sym1 First expression indicating symmetrized indexes
        \param sym2 Second expression indicating symmetrized indexes
        \param subexpr Sub-expression.
     **/
    symm2_core(const letter_expr<M> &sym1, const letter_expr<M> &sym2,
        const expr<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~symm2_core() { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new symm2_core(*this);
    }

    /** \brief Returns the first set of symmetrized indexes
     **/
    const letter_expr<M> &get_sym1() const {
        return m_sym1;
    }

    /** \brief Returns the second set of symmetrized indexes
     **/
    const letter_expr<M> &get_sym2() const {
        return m_sym2;
    }

    /** \brief Returns the sub-expression
     **/
    expr<N, T> &get_sub_expr() {
        return m_subexpr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr<N, T> &get_sub_expr() const {
        return m_subexpr;
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


/** \brief Evaluating container for the symmetrization over two sets of
        indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in the set.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, bool Sym, typename T>
class symm2_eval : public eval_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

    //!    Expression core type
    typedef symm2_core<N, M, Sym, T, SubCore> core_t;

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
    permutation<N> m_perm; //!< Permutation for symmetrization
    btod_symmetrize2<N> *m_op; //!< Symmetrization operation
    arg<N, T, oper_tag> *m_arg; //!< Argument

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    symm2_eval(const expr<N, T> &e, const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~symm2_eval();

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


template<size_t N, size_t M, bool Sym, typename T>
const char symm2_core<N, M, Sym, T>::k_clazz[] = "symm2_core<N, M, Sym, T>";


template<size_t N, size_t M, bool Sym, typename T>
symm2_core<N, M, Sym, T>::symm2_core(const letter_expr<M> &sym1,
    const letter_expr<M> &sym2, const expr<N, T> &expr) :

    m_sym1(sym1), m_sym2(sym2), m_expr(expr) {

    static const char method[] = "symm2_core(const letter_expr<M>&, "
        "const letter_expr<M>&, const expr<N, T>&)";

    for(size_t i = 0; i < M; i++) {
        if(sym2.contains(sym1.letter_at(i))) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Symmetrized indexes must be different.");
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l1 = m_sym1.letter_at(i);
        const letter &l2 = m_sym2.letter_at(i);
        if(!m_expr.contains(l1) || !m_expr.contains(l2)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Symmetrized index is absent from the sub-expression.");
        }
    }
}


template<size_t N, size_t M, bool Sym, typename T>
const char symm2_eval<N, M, Sym, T>::k_clazz[] = "symm2_eval<N, M, Sym, T>";


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
template<int Dummy>
struct symm2_eval<N, M, Sym, T, SubCore>::narg<oper_tag, Dummy> {
    static const size_t k_narg = 1;
};


template<size_t N, size_t M, bool Sym, typename T>
symm2_eval<N, M, Sym, T>::symm2_eval(
    const expr<N, T> &expr,
    const letter_expr<N> &label) :

    m_sub_expr(expr.get_core().get_sub_expr()),
    m_sub_eval_cont(m_sub_expr, label),
    m_sub_eval(m_sub_expr, m_sub_eval_cont),
    m_op(0), m_arg(0) {

    for(size_t i = 0; i < M; i++) {
        size_t i1 = label.index_of(
            expr.get_core().get_sym1().letter_at(i));
        size_t i2 = label.index_of(
            expr.get_core().get_sym2().letter_at(i));
        m_perm.permute(i1, i2);
    }
}


template<size_t N, size_t M, bool Sym, typename T>
symm2_eval<N, M, Sym, T>::~symm2_eval() {

    destroy_arg();
}


template<size_t N, size_t M, bool Sym, typename T>
void symm2_eval<N, M, Sym, T>::prepare() {

    m_sub_eval_cont.prepare();
    create_arg();
}


template<size_t N, size_t M, bool Sym, typename T>
void symm2_eval<N, M, Sym, T>::clean() {

    destroy_arg();
    m_sub_eval_cont.clean();
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
void symm2_eval<N, M, Sym, T, SubCore>::create_arg() {

    destroy_arg();
    m_op = new btod_symmetrize2<N>(m_sub_eval.get_bto(), m_perm, Sym);
    m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
void symm2_eval<N, M, Sym, T, SubCore>::destroy_arg() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
template<typename Tag>
arg<N, T, Tag> symm2_eval<N, M, Sym, T, SubCore>::get_arg(const Tag &tag,
    size_t i) const {

    static const char *method = "get_arg(const Tag&, size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, size_t M, bool Sym, typename T, typename SubCore>
arg<N, T, oper_tag> symm2_eval<N, M, Sym, T, SubCore>::get_arg(
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

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
