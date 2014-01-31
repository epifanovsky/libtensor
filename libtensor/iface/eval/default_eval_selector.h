#ifndef LIBTENSOR_IFACE_DEFAULT_EVAL_SELECTOR_H
#define LIBTENSOR_IFACE_DEFAULT_EVAL_SELECTOR_H

#include "eval_selector_i.h"

namespace libtensor {
namespace iface {


/** \brief Default evaluator selector

    This evaluator selector chooses the first evaluator that can evaluate
    a given expression.

    \ingroup libtensor_iface
 **/
class default_eval_selector : public eval_selector_i {
private:
    const expr::expr_tree &m_expr; //!< Expression to be evaluated
    const eval_i *m_eval; //!< Evaluator

public:
    /** \brief Initializes the selector
     **/
    default_eval_selector(const expr::expr_tree &e) : m_expr(e), m_eval(0) { }

    /** \brief Virtual destructor
     **/
    virtual ~default_eval_selector();

    /** \brief Returns the selected evaluator or throws an exception
     **/
    const eval_i &get_selected() const;

    /** \brief Tests an evaluator for suitability
     **/
    virtual void try_evaluator(const eval_i &e);

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_DEFAULT_EVAL_SELECTOR_H
