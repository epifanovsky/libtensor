#ifndef LIBTENSOR_LABELED_BTENSOR_IMPL_H
#define LIBTENSOR_LABELED_BTENSOR_IMPL_H

#include <libtensor/exception.h>
#include "labeled_btensor.h"
#include "expr/expr_rhs.h"
#include "expr/eval.h"
#include "ident/ident_core.h"

namespace libtensor {

template<size_t N, typename T>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    const labeled_btensor_expr::expr_rhs<N, T> &rhs) {

    labeled_btensor_expr::eval<N, T>(rhs, *this).evaluate();
    return *this;
}

template<size_t N, typename T>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    const labeled_btensor<N, T, false> &rhs) {

    labeled_btensor_expr::expr_rhs<N, T> e(
            new labeled_btensor_expr::ident_core<N, T, false>(rhs));
    labeled_btensor_expr::eval<N, T>(e, *this).evaluate();
    return *this;
}


template<size_t N, typename T>
labeled_btensor<N, T, true> &labeled_btensor<N, T, true>::operator=(
    const labeled_btensor<N, T, true> &rhs) {

    labeled_btensor_expr::expr_rhs<N, T> e(
            new labeled_btensor_expr::ident_core<N, T, true>(rhs));
    labeled_btensor_expr::eval<N, T>(e, *this).evaluate();
    return *this;
}


} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_IMPL_H
