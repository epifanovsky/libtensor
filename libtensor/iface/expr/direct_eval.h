#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_EVAL_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_EVAL_H

#include <libtensor/exception.h>
#include "../direct_btensor.h"
#include "expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/** \brief Evaluates an expression on the fly
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Core Expression core type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class direct_eval {
private:
    direct_btensor<N, T> m_bt; //!< Direct block tensor

public:
    //!    \name Construction and destruction
    //@{

    direct_eval(const expr<N, T> &e, const letter_expr<N> &label) :
        m_bt(label, e) { }

    ~direct_eval() { }

    //@}

    //!    \name Evaluation
    //@{

    /** \brief Evaluates the expression
     **/
    void evaluate() { }

    /** \brief Cleans up the temporary block %tensor
     **/
    void clean() { }

    /** \brief Returns the block %tensor
     **/
    btensor_rd_i<N, T> &get_btensor() {
        return m_bt;
    }

    //@}

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRECT_EVAL_H
