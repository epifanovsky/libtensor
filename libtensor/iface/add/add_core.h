#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H

#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Addition operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class add_core : public expr_core_i<N, T> {
private:
    expr<N, T> m_expr_l; //!< Left expression
    expr<N, T> m_expr_r; //!< Right expression

public:
    /** \brief Initializes the core with left and right expressions
     **/
    add_core(const expr<N, T> &expr_l, const expr<N, T> &expr_r) :
        m_expr_l(expr_l), m_expr_r(expr_r)
    { }

    /** \brief Clones this object using new
     **/
    expr_core_i<N, T> *clone() const {
        return new add_core(*this);
    }

    /** \brief Returns the left expression
     **/
    const expr<N, T> &get_expr_l() {
        return m_expr_l;
    }

    /** \brief Returns the right expression
     **/
    const expr<N, T> &get_expr_r() {
        return m_expr_r;
    }

    /** \brief Creates evaluation container using new
     **/
    virtual eval_container_i<N, T> *create_container(
        const letter_expr<N> &label) const;

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr_l.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr_l.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr_l.letter_at(i);
    }

};


/** \brief Evaluates the addition expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class add_eval : public eval_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

private:
    expr<N, T> m_expr; //!< Addition expression
    add_core<N, T> &m_core; //!< Expression core
    eval_container_i<N, T> *m_cont_l; //!< Left evaluating container
    eval_container_i<N, T> *m_cont_r; //!< Right evaluating container

public:
    //! \name Construction and destruction
    //@{

    add_eval(expr<N, T> &e, const letter_expr<N> &label);

    virtual ~add_eval();

    //@}


    //! \name Evaluation
    //@{

    void prepare();

    void clean();

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const;

    //@}
};


template<size_t N, typename T>
const char add_eval<N, T>::k_clazz[] = "add_eval<N, T>";


template<size_t N, typename T>
add_eval<N, T>::add_eval(expr<N, T> &e, const letter_expr<N> &label) :

    m_expr(e),
    m_core(dynamic_cast< add_core<N, T>& >(m_expr.get_core())),
    m_cont_l(m_core.get_expr_l().get_core().create_container(label)),
    m_cont_r(m_core.get_expr_r().get_core().create_container(label)) {

}


template<size_t N, typename T>
add_eval<N, T>::~add_eval() {

    delete m_cont_l;
    delete m_cont_r;
}


template<size_t N, typename T>
void add_eval<N, T>::prepare() {

    m_cont_l->prepare();
    m_cont_r->prepare();
}


template<size_t N, typename T>
void add_eval<N, T>::clean() {

    m_cont_l->clean();
    m_cont_r->clean();
}


template<size_t N, typename T>
template<typename Tag>
arg<N, T, Tag> add_eval<N, T>::get_arg(const Tag &tag, size_t i) const {

    static const char method[] = "get_arg(const Tag&, size_t)";
    if(i > narg<Tag>::k_narg) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    const size_t narg_l = eval_container_l_t::template narg<Tag>::k_narg;
    const size_t narg_r = eval_container_r_t::template narg<Tag>::k_narg;

    return (narg_l > 0 && narg_l > i) ?
        m_cont_l.get_arg(tag, i) : m_cont_r.get_arg(tag, i - narg_l);
}


template<size_t N, typename T>
eval_container_i<N, T> *add_core<N, T>::create_container(
    const letter_expr<N> &label) const {

    return new add_eval<N, T>(*this, label);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ADD_CORE_H
