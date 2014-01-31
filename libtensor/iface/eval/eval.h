#ifndef LIBTENSOR_IFACE_EVAL_H
#define LIBTENSOR_IFACE_EVAL_H

#include "eval_selector_i.h"

namespace libtensor {
namespace iface {


/** \brief Generic expression evaluator

    \ingroup libtensor_iface
 **/
class eval {
public:
    /** \brief Evaluates the given expression with a given evaluator selector
     **/
    void evaluate(const expr::expr_tree &e, eval_selector_i &es) const;

    /** \brief Evaluates the given expression with a default evaluator selector
     **/
    void evaluate(const expr::expr_tree &e) const;

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EVAL_H
