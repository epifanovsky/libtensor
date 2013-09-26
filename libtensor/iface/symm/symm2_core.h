#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H

#include <libtensor/exception.h>
#include <libtensor/block_tensor/btod_symmetrize2.h>
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

private:
    letter_expr<M> m_sym1; //!< First set of symmetrized indexes
    letter_expr<M> m_sym2; //!< Second set of symmetrized indexes
    expr_rhs<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param sym1 First expression indicating symmetrized indexes
        \param sym2 Second expression indicating symmetrized indexes
        \param subexpr Sub-expression.
     **/
    symm2_core(
        const letter_expr<M> &sym1,
        const letter_expr<M> &sym2,
        const expr_rhs<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~symm2_core() { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new symm2_core<N, M, Sym, T>(*this);
    }

    /** \brief Creates an evaluation container using new, caller responsible
            to call delete
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

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
    expr_rhs<N, T> &get_sub_expr() {
        return m_subexpr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr_rhs<N, T> &get_sub_expr() const {
        return m_subexpr;
    }

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    virtual bool contains(const letter &let) const {
        return m_subexpr.get_core().contains(let);
    }

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    virtual size_t index_of(const letter &let) const {
        return m_subexpr.get_core().index_of(let);
    }

    /** \brief Returns the letter at a given position in the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    virtual const letter &letter_at(size_t i) const {
        return m_subexpr.get_core().letter_at(i);
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
class symm2_eval : public eval_container_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    symm2_core<N, M, Sym, T> m_core; //!< Expression core
    std::auto_ptr< eval_container_i<N, T> > m_sub_eval_cont; //!< Evaluation of the sub-expression
    evalfunctor<N, T> m_functor;
    letter_expr<N> m_label;

    btod_symmetrize2<N> *m_op; //!< Symmetrization operation
//    arg<N, T, oper_tag> *m_arg; //!< Argument

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    symm2_eval(
        const symm2_core<N, M, Sym, T> &core,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~symm2_eval();

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
//   virtual arg<N, T, tensor_tag> get_tensor_arg(size_t i);

   /** \brief Returns operation arguments
       \param i Argument number (0 is the only valid value).
    **/
//    virtual arg<N, T, oper_tag> get_oper_arg(size_t i);
};


template<size_t N, size_t M, bool Sym, typename T>
const char symm2_core<N, M, Sym, T>::k_clazz[] = "symm2_core<N, M, Sym, T>";


template<size_t N, size_t M, bool Sym, typename T>
symm2_core<N, M, Sym, T>::symm2_core(
    const letter_expr<M> &sym1,
    const letter_expr<M> &sym2,
    const expr_rhs<N, T> &subexpr) :

    m_sym1(sym1), m_sym2(sym2), m_subexpr(subexpr) {

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
        if(!m_subexpr.get_core().contains(l1) ||
                !m_subexpr.get_core().contains(l2)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Symmetrized index is absent from the sub-expression.");
        }
    }
}


template<size_t N, size_t M, bool Sym, typename T>
eval_container_i<N, T> *symm2_core<N, M, Sym, T>::create_container(
    const letter_expr<N> &label) const {

    return new symm2_eval<N, M, Sym, T>(*this, label);
}


template<size_t N, size_t M, bool Sym, typename T>
const char symm2_eval<N, M, Sym, T>::k_clazz[] = "symm2_eval<N, M, Sym, T>";


template<size_t N, size_t M, bool Sym, typename T>
symm2_eval<N, M, Sym, T>::symm2_eval(
    const symm2_core<N, M, Sym, T> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_sub_eval_cont(m_core.get_sub_expr().get_core().create_container(label)),
    m_functor(m_core.get_sub_expr(), *m_sub_eval_cont),
    m_label(label),
    m_op(0)/*, m_arg(0)*/ {

}


template<size_t N, size_t M, bool Sym, typename T>
symm2_eval<N, M, Sym, T>::~symm2_eval() {

//    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


template<size_t N, size_t M, bool Sym, typename T>
void symm2_eval<N, M, Sym, T>::prepare() {

    m_sub_eval_cont->prepare();
//    delete m_arg;
    delete m_op;

    permutation<N> perm;
    for(size_t i = 0; i < M; i++) {
        size_t i1 = m_label.index_of(m_core.get_sym1().letter_at(i));
        size_t i2 = m_label.index_of(m_core.get_sym2().letter_at(i));
        perm.permute(i1, i2);
    }

    m_op = new btod_symmetrize2<N>(m_functor.get_bto(), perm, Sym);
//    m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);

}


template<size_t N, size_t M, bool Sym, typename T>
void symm2_eval<N, M, Sym, T>::clean() {

    m_sub_eval_cont->clean();
//    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


//template<size_t N, size_t M, bool Sym, typename T>
//arg<N, T, tensor_tag> symm2_eval<N, M, Sym, T>::get_tensor_arg(size_t i) {
//
//    static const char *method = "get_tensor_arg(size_t)";
//
//    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
//        "Invalid method.");
//}
//
//
//template<size_t N, size_t M, bool Sym, typename T>
//arg<N, T, oper_tag> symm2_eval<N, M, Sym, T>::get_oper_arg(size_t i) {
//
//    static const char *method = "get_oper_oper_arg(size_t)";
//
//    if(i != 0) {
//        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
//            "Argument index is out of bounds.");
//    }
//
//    return *m_arg;
//}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
