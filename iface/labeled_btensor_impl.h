#ifndef LIBTENSOR_LABELED_BTENSOR_IMPL_H
#define LIBTENSOR_LABELED_BTENSOR_IMPL_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_eval.h"
#include "labeled_btensor_expr.h"
#include "labeled_btensor_expr_ident.h"

namespace libtensor {

template<size_t N, typename T, typename Label> template<typename Expr>
labeled_btensor<N, T, true, Label>
labeled_btensor<N, T, true, Label>::operator=(
	const labeled_btensor_expr::expr<N, T, Expr> &rhs) throw(exception) {

	labeled_btensor_expr::eval<N, T, Expr, Label> eval(rhs, *this);
	eval.evaluate();
	return *this;
}

template<size_t N, typename T, typename Label>
template<bool AssignableR, typename LabelR>
labeled_btensor<N, T, true, Label>
labeled_btensor<N, T, true, Label>::operator=(
	labeled_btensor<N, T, AssignableR, LabelR> rhs) throw(exception) {

	typedef labeled_btensor_expr::core_ident<N, T, AssignableR, LabelR> id_t;
	typedef labeled_btensor_expr::expr<N, T, id_t> expr_t;
	id_t id(rhs);
	expr_t op(id);
	labeled_btensor_expr::eval<N, T, id_t, Label> eval(op, *this);
	eval.evaluate();
	return *this;
}

template<size_t N, typename T, typename Label>
labeled_btensor<N, T, true, Label>
labeled_btensor<N, T, true, Label>::operator=(
	labeled_btensor<N, T, true, Label> rhs) throw(exception) {

	typedef labeled_btensor_expr::core_ident<N, T, true, Label> id_t;
	typedef labeled_btensor_expr::expr<N, T, id_t> expr_t;
	id_t id(rhs);
	expr_t op(id);
	labeled_btensor_expr::eval<N, T, id_t, Label> eval(op, *this);
	eval.evaluate();
	return *this;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_IMPL_H
