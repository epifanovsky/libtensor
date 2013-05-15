#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Expression core that scales an underlying expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class scale_core : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    T m_coeff; //!< Scaling coefficient
    expr<N, T> m_expr; //!< Unscaled expression

public:
    /** \brief Constructs the scaling expression using a coefficient
            and the underlying unscaled expression
     **/
    scale_core(const T &coeff, const expr<N, T> &subexpr) :
        m_coeff(coeff), m_expr(subexpr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~scale_core() { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new scale_core(*this);
    }

    /** \brief Returns the unscaled expression
     **/
    expr<N, T> &get_unscaled_expr() {
        return m_expr;
    }

    /** \brief Returns the unscaled expression (const version)
     **/
    const expr<N, T> &get_unscaled_expr() const {
        return m_expr;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_coeff() {
        return m_coeff;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr.letter_at(i);
    }

};


/** \brief Evaluates a scaled expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class scale_eval : public eval_i<N, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    //!    Scaling expression core type
    typedef core_scale<N, T, Expr> core_t;

    //!    Scaled expression type
    typedef expr<N, T, core_t> expression_t;

    //!    Unscaled expression evaluating container type
    typedef typename Expr::eval_container_t unscaled_eval_container_t;

    //!    Number of arguments in the expression
    template<typename Tag>
    struct narg {
        static const size_t k_narg =
            unscaled_eval_container_t::template narg<Tag>::k_narg;
    };

private:
    expr<N, T> m_expr; //!< Expression
    scale_core<N, T> &m_core; //!< Expression core
    eval_container_i<N, T> *m_unscaled_cont; //!< Original expression

public:
    /** \brief Constructs the evaluating container
     **/
    scale_eval(expr<N, T> &e, const letter_expr<N> &label);

    /** \brief Virtual destructor
     **/
    virtual ~scale_eval();

    /** \brief Evaluates sub-expressions into temporary tensors
     **/
    virtual void prepare();

    /** \brief Cleans up temporary tensors
     **/
    virtual void clean();

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

};


template<size_t N, typename T>
const char scale_eval<N, T>::k_clazz[] = "scale_eval<N, T>";


template<size_t N, typename T>
scale_eval<N, T>::scale_eval(expr<N, T> &e, const letter_expr<N> &label) :

    m_expr(e),
    m_core(dynamic_cast< scale_core<N, T>& >(m_expr.get_core())),
    m_unscaled_cont(m_core.get_unscaled_expr().get_core().
        create_container(label)) {

}


template<size_t N, typename T>
scale_eval<N, T>::~scale_eval() {

    delete m_unscaled_cont;
}


template<size_t N, typename T>
void scale_eval<N, T>::prepare() {

    m_unscaled_cont->prepare();
}


template<size_t N, typename T>
void scale_eval<N, T>::clean() {

    m_unscaled_cont->clean();
}


template<size_t N, typename T>
template<typename Tag>
inline arg<N, T, Tag> scale_eval<N, T>::get_arg(const Tag &tag, size_t i)
    const throw(exception) {

    arg<N, T, Tag> argument = m_unscaled_cont.get_arg(tag, i);
    argument.scale(m_expr.get_core().get_coeff());
    return argument;
}


template<size_t N, typename T>
eval_container_i<N, T> *scale_core<N, T>::create_container(
    const letter_expr<N> &label) const {

    return new scale_eval<N, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SCALE_CORE_H
