#ifndef LIBTENSOR_DOT_PRODUCT_H
#define LIBTENSOR_DOT_PRODUCT_H

#include "defs.h"
#include "exception.h"
#include "btod_dotprod.h"
#include "labeled_btensor.h"
#include "letter.h"
#include "letter_expr.h"
#include "permutation_builder.h"

namespace libtensor {

/**	\brief Dot product (%tensor + %tensor)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable1, typename Label1,
	bool Assignable2, typename Label2>
double dot_product(labeled_btensor<N, T, Assignable1, Label1> bt1,
	labeled_btensor<N, T, Assignable2, Label2> bt2) {

	size_t seq1[N], seq2[N];
	for(size_t i = 0; i < N; i++) {
		seq1[i] = i;
		seq2[i] = bt2.index_of(bt1.letter_at(i));
	}
	permutation<N> perma;
	permutation_builder<N> permb(seq1, seq2);
	btod_dotprod<N> op(
		bt1.get_btensor(), perma, bt2.get_btensor(), permb.get_perm());
	return op.calculate();
}

/**	\brief Dot product (%tensor + expression)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Assignable1, typename Label1,
	typename Expr2>
double dot_product(labeled_btensor<N, T, Assignable1, Label1> bt1,
	labeled_btensor_expr<N, T, Expr2> bt2) {

	return 0.0;
}

/**	\brief Dot product (expression + %tensor)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr1, bool Assignable2,
	typename Label2>
double dot_product(labeled_btensor_expr<N, T, Expr1> bt1,
	labeled_btensor<N, T, Assignable2, Label2> bt2) {

	return 0.0;
}

/**	\brief Dot product (expression + expression)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr1, typename Expr2>
double dot_product(labeled_btensor_expr<N, T, Expr1> bt1,
	labeled_btensor_expr<N, T, Expr2> bt2) {

	return 0.0;
}

} // namespace libtensor

#endif // LIBTENSOR_DOT_PRODUCT_H
