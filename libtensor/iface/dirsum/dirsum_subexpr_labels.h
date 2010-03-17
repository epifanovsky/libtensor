#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABELS_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABELS_H

#include "core_dirsum.h"
#include "dirsum_subexpr_label_builder.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Labels for sub-expressions in dirsum

	\ingroup libtensor_iface
 **/
template<size_t N, size_t M, typename T, typename E1, typename E2>
class dirsum_subexpr_labels {
public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

	//!	Contraction expression core type
	typedef core_dirsum<N, M, T, E1, E2> core_t;

	//!	Contraction expression type
	typedef expr<k_orderc, T, core_t> expression_t;

private:
	//!	Label builder for the first sub-expression
	dirsum_subexpr_label_builder<N, M> m_bld_a;

	//!	Label builder for the second sub-expression
	dirsum_subexpr_label_builder<M, N> m_bld_b;

public:
	/**	\brief Initializes the object using a contract expression and
			a result label
	 **/
	dirsum_subexpr_labels(const expression_t &expr,
		const letter_expr<k_orderc> &label_c);

	/**	\brief Returns the label for the first sub-expression
	 **/
	const letter_expr<N> &get_label_a() const {
		return m_bld_a.get_label();
	}

	/**	\brief Returns the label for the second sub-expression
	 **/
	const letter_expr<M> &get_label_b() const {
		return m_bld_b.get_label();
	}
};


template<size_t N, size_t M, typename T, typename E1, typename E2>
dirsum_subexpr_labels<N, M, T, E1, E2>::dirsum_subexpr_labels(
	const expression_t &expr, const letter_expr<k_orderc> &label_c) :

	m_bld_a(label_c, expr.get_core().get_expr_1()),
	m_bld_b(label_c, expr.get_core().get_expr_2()) {

}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_SUBEXPR_LABELS_H
