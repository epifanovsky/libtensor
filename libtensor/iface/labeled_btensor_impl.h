#ifndef LIBTENSOR_LABELED_BTENSOR_IMPL_H
#define LIBTENSOR_LABELED_BTENSOR_IMPL_H

#include <libtensor/exception.h>
#include "labeled_btensor.h"
#include "expr/expr.h"
#include "expr/eval.h"
#include "ident/ident_core.h"

namespace libtensor {

template<size_t N, typename T>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    const labeled_btensor_expr::expr<N, T> rhs) {

    labeled_btensor_expr::eval<N, T>(rhs, *this).evaluate();
    return *this;
}

template<size_t N, typename T>
template<bool AssignableR>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    labeled_btensor<N, T, AssignableR> rhs) {

    expr<N, T> e(ident_core<N, T, AssignableR>(rhs));
    labeled_btensor_expr::eval<N, T>(e, *this).evaluate();
    return *this;
}

template<size_t N, typename T>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    labeled_btensor<N, T, true> rhs) {

    expr<N, T> e(ident_core<N, T, true>(rhs));
    labeled_btensor_expr::eval<N, T>(e, *this).evaluate();
    eval.evaluate();
    return *this;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_IMPL_H
