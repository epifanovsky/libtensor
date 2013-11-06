#ifndef LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
#define LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H

#include <libtensor/iface/expr_tree.h>

namespace libtensor {
namespace iface {


/** \brief Processor of evaluation plan for btensor result type (double)

    \ingroup libtensor_iface
 **/
template<>
class eval_btensor<double> {
public:
    enum {
        Nmax = 8
    };

public:
    /** \brief Evaluates an expression tree
     **/
    void evaluate(expr_tree &tree);

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
