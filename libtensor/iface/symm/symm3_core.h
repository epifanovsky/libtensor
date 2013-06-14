#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H

#include <libtensor/exception.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
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
    symm3_core(
        const letter &l1,
        const letter &l2,
        const letter &l3,
        const expr<N, T> &subexpr);

    /** \brief Virtual destructor
     **/
    virtual ~symm3_core() { }

    /** \brief Clones this object using new
     **/
    virtual expr_core_i<N, T> *clone() const {
        return new symm3_core<N, Sym, T>(*this);
    }

    /** \brief Creates an evaluation container using new, caller responsible
            to call delete
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

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


/** \brief Evaluating container for the symmetrization three indexes
    \tparam N Tensor order.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T>
class symm3_eval : public eval_container_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name


private:
    symm3_core<N, Sym, T> m_core; //!< Sub-expression
    std::auto_ptr< eval_container_i<N, T> > m_sub_eval_cont; //!< Evaluation of the sub-expression
    letter_expr<N> m_label;

    btod_symmetrize3<N> *m_op; //!< Symmetrization operation
    arg<N, T, oper_tag> *m_arg; //!< Argument

public:
    /** \brief Initializes the container with given expression and
            result recipient
     **/
    symm3_eval(
        const symm3_core<N, Sym, T> &e,
        const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~symm3_eval();

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


template<size_t N, bool Sym, typename T>
const char symm3_core<N, Sym, T>::k_clazz[] = "symm3_core<N, Sym, T>";


template<size_t N, bool Sym, typename T>
symm3_core<N, Sym, T>::symm3_core(
    const letter &l1,
    const letter &l2,
    const letter &l3,
    const expr<N, T> &subexpr) :

    m_l1(l1), m_l2(l2), m_l3(l3), m_subexpr(subexpr) {

    static const char method[] = "symm3_core(const letter&, "
        "const letter&, const letter&, const expr<N, T>&)";

    if(m_l1 == m_l2 || m_l1 == m_l3 || m_l2 == m_l3) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Symmetrized indexes must be different.");
    }
}


template<size_t N, bool Sym, typename T>
eval_container_i<N, T> *symm3_core<N, Sym, T>::create_container(
    const letter_expr<N> &label) const {

    return new symm3_eval<N, Sym, T>(*this, label);
}


template<size_t N, bool Sym, typename T>
const char symm3_eval<N, Sym, T>::k_clazz[] = "symm3_eval<N, Sym, T>";


template<size_t N, bool Sym, typename T>
symm3_eval<N, Sym, T>::symm3_eval(
    const symm3_core<N, Sym, T> &core,
    const letter_expr<N> &label) :

    m_core(core),
    m_sub_eval_cont(m_core.get_sub_expr().get_core().create_container(label)),
    m_label(label),
    m_op(0), m_arg(0) {

}


template<size_t N, bool Sym, typename T>
symm3_eval<N, Sym, T>::~symm3_eval() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T>::prepare() {

    m_sub_eval_cont->prepare();

    size_t i1 = m_label.index_of(m_core.get_l1());
    size_t i2 = m_label.index_of(m_core.get_l2());
    size_t i3 = m_label.index_of(m_core.get_l3());

    if (m_arg != 0) delete m_arg;
    if (m_op != 0) delete m_op;

    m_op = new btod_symmetrize3<N>(
            m_sub_eval_cont->get_oper_arg(0).get_operation(), i1, i2, i3, Sym);
    m_arg = new arg<N, T, oper_tag>(*m_op, 1.0);
}


template<size_t N, bool Sym, typename T>
void symm3_eval<N, Sym, T>::clean() {

    delete m_arg; m_arg = 0;
    delete m_op; m_op = 0;
    m_sub_eval_cont->clean();
}


template<size_t N, bool Sym, typename T>
arg<N, T, tensor_tag> symm3_eval<N, Sym, T>::get_tensor_arg(size_t i) {

    static const char *method = "get_tensor_arg(size_t)";

    throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
        "Invalid method.");
}


template<size_t N, bool Sym, typename T>
arg<N, T, oper_tag> symm3_eval<N, Sym, T>::get_oper_arg(size_t i) {

    static const char *method = "get_arg(size_t)";
    if(i != 0) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }
    return *m_arg;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
