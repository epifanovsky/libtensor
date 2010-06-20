#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_OPERATOR_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_OPERATOR_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../../btod/btod_trace.h"
#include "../labeled_btensor.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "trace_subexpr_label_builder.h"

namespace libtensor {

namespace labeled_btensor_expr {


/**	\brief Trace of a matrix

	\ingroup libtensor_btensor_expr
 **/
template<typename T, bool A>
double trace(
	const letter &l1,
	const letter &l2,
	labeled_btensor<2, T, A> bt) {

	size_t seq1[2], seq2[2];
	seq1[0] = 0;
	seq2[0] = bt.index_of(l1);
	seq1[1] = 1;
	seq2[1] = bt.index_of(l2);

	permutation_builder<2> pb(seq1, seq2);
	return btod_trace<1>(bt.get_btensor(), pb.get_perm()).calculate();
}


/**	\brief Trace of a %tensor

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool A>
double trace(
	letter_expr<N> le1,
	letter_expr<N> le2,
	labeled_btensor<2 * N, T, A> bt) {

	size_t seq1[2 * N], seq2[2 * N];
	for(size_t i = 0; i < N; i++) {
		seq1[i] = i;
		seq2[i] = bt.index_of(le1.letter_at(i));
		seq1[N + i] = N + i;
		seq2[N + i] = bt.index_of(le2.letter_at(i));
	}

	permutation_builder<2 * N> pb(seq1, seq2);
	return btod_trace<N>(bt.get_btensor(), pb.get_perm()).calculate();
}


/**	\brief Trace of a matrix expression

	\ingroup libtensor_btensor_expr
 **/
template<typename T, typename E>
double trace(
	const letter &l1,
	const letter &l2,
	expr<2, T, E> expr) {

	letter_expr<2> le(l1|l2);
	anon_eval<2, T, E> eval(expr, le);
	eval.evaluate();
	return btod_trace<1>(eval.get_btensor()).calculate();
}


/**	\brief Trace of a %tensor expression

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename E>
double trace(
	letter_expr<N> le1,
	letter_expr<N> le2,
	expr<2 * N, T, E> expr) {

	trace_subexpr_label_builder<N> lb(le1, le2);
	anon_eval<2 * N, T, E> eval(expr, lb.get_label());
	eval.evaluate();
	return btod_trace<N>(eval.get_btensor()).calculate();
}


} // namespace labeled_btensor_expr

using labeled_btensor_expr::trace;

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_TRACE_OPERATOR_H
