#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_ADD_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_ADD_H

#include "../expr/eval_i.h"
#include "core_add.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluates the addition expression
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam ExprL LHS expression type.
    \tparam ExprR RHS expression type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename ExprL, typename ExprR>
class eval_add : public eval_i<N, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    //!    Addition expression core type
    typedef core_add<N, T, ExprL, ExprR> core_t;

    //!    Addition expression type
    typedef expr<N, T, core_t> expression_t;

    //!    Evaluating container type for the left expression
    typedef typename ExprL::eval_container_t eval_container_l_t;

    //!    Evaluating container type for the right expression
    typedef typename ExprR::eval_container_t eval_container_r_t;

    //!    Number of arguments in the expression
    template<typename Tag>
    struct narg {
        static const size_t k_narg =
            eval_container_l_t::template narg<Tag>::k_narg +
            eval_container_r_t::template narg<Tag>::k_narg;
    };

private:
    expression_t &m_expr; //!< Addition expression
    eval_container_l_t m_cont_l; //!< Left evaluating container
    eval_container_r_t m_cont_r; //!< Right evaluating container

public:
    eval_add(expression_t &expr, const letter_expr<N> &label)
        throw(exception);

    virtual ~eval_add() { }

    //!    \name Evaluation
    //@{

    void prepare();

    void clean();

    template<typename Tag>
    arg<N, T, Tag> get_arg(const Tag &tag, size_t i) const throw(exception);

    //@}
};


template<size_t N, typename T, typename ExprL, typename ExprR>
const char *eval_add<N, T, ExprL, ExprR>::k_clazz =
    "eval_add<N, T, ExprL, ExprR>";


template<size_t N, typename T, typename ExprL, typename ExprR>
eval_add<N, T, ExprL, ExprR>::eval_add(
    expression_t &expr, const letter_expr<N> &label) throw(exception) :
        m_expr(expr),
        m_cont_l(expr.get_core().get_expr_l(), label),
        m_cont_r(expr.get_core().get_expr_r(), label) {

}


template<size_t N, typename T, typename ExprL, typename ExprR>
void eval_add<N, T, ExprL, ExprR>::prepare() {

    m_cont_l.prepare();
    m_cont_r.prepare();
}


template<size_t N, typename T, typename ExprL, typename ExprR>
void eval_add<N, T, ExprL, ExprR>::clean() {

    m_cont_l.clean();
    m_cont_r.clean();
}


template<size_t N, typename T, typename ExprL, typename ExprR>
template<typename Tag>
arg<N, T, Tag> eval_add<N, T, ExprL, ExprR>::get_arg(
    const Tag &tag, size_t i) const throw(exception) {

    if(i > narg<Tag>::k_narg) {
        throw out_of_bounds(g_ns, k_clazz,
            "get_arg(const Tag&, size_t)", __FILE__, __LINE__,
            "Argument index is out of bounds.");
    }

    const size_t narg_l = eval_container_l_t::template narg<Tag>::k_narg;
    const size_t narg_r = eval_container_r_t::template narg<Tag>::k_narg;

    return (narg_l > 0 && narg_l > i) ?
        m_cont_l.get_arg(tag, i) : m_cont_r.get_arg(tag, i - narg_l);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVAL_ADD_H
