#ifndef LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
#define LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H

#include <libtensor/expr/expr_tree.h>
#include <libtensor/iface/eval_i.h>

namespace libtensor {
namespace iface {


/** \brief Processor of evaluation plan for btensor result type (double)

    \ingroup libtensor_iface
 **/
template<>
class eval_btensor<double> : public eval_i {
public:
    enum {
        Nmax = 8
    };

public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_btensor<double>();

    /** \brief Checks if this evaluator can handle the given expression
     **/
    virtual bool can_evaluate(const expr::expr_tree &e) const;

    /** \brief Evaluates an expression tree
     **/
    virtual void evaluate(const expr::expr_tree &tree) const;

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
