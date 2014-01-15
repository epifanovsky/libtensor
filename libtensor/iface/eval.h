#ifndef LIBTENSOR_IFACE_EVAL_H
#define LIBTENSOR_IFACE_EVAL_H

#include "eval_i.h"

namespace libtensor {
namespace iface {


/** \brief Generic evaluator

    \ingroup libtensor_iface
 **/
class eval : public eval_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~eval() { }

    /** \brief Returns true if this evaluator can handle the given expression
     **/
    virtual bool can_evaluate(const expr::expr_tree &e) const;

    /** \brief Evaluates the given expression
     **/
    virtual void evaluate(const expr::expr_tree &e) const;

};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EVAL_H
