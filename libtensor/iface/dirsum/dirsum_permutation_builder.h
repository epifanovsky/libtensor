#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_PERMUTATION_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_PERMUTATION_BUILDER_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Builds the permutation object using labels
	\tparam N Order of the first %tensor (A).
	\tparam M Order of the second %tensor (B).

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M>
class dirsum_permutation_builder {
public:
	static const size_t k_ordera = N; //!< Order of the first %tensor
	static const size_t k_orderb = M; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	permutation<N + M> m_perm;

public:
	dirsum_permutation_builder(
		const letter_expr<k_ordera> &label_a,
		const permutation<k_ordera> &perma,
		const letter_expr<k_orderb> &label_b,
		const permutation<k_orderb> &permb,
		const letter_expr<k_orderc> &label_c);

	const permutation<N + M> &get_perm() const { return m_perm; }

private:
	static permutation<N + M> mk_perm(
		const letter_expr<k_ordera> &label_a,
		const permutation<k_ordera> &perma,
		const letter_expr<k_orderb> &label_b,
		const permutation<k_orderb> &permb,
		const letter_expr<k_orderc> &label_c);
};


template<size_t N, size_t M>
dirsum_permutation_builder<N, M>::dirsum_permutation_builder(
	const letter_expr<k_ordera> &label_a,
	const permutation<k_ordera> &perma,
	const letter_expr<k_orderb> &label_b,
	const permutation<k_orderb> &permb,
	const letter_expr<k_orderc> &label_c) :

	m_perm(mk_perm(label_a, perma, label_b, permb, label_c)) {

}


template<size_t N, size_t M>
permutation<N + M> dirsum_permutation_builder<N, M>::mk_perm(
	const letter_expr<k_ordera> &label_a,
	const permutation<k_ordera> &perma,
	const letter_expr<k_orderb> &label_b,
	const permutation<k_orderb> &permb,
	const letter_expr<k_orderc> &label_c) {

	sequence<k_ordera, size_t> seq2a(0);
	sequence<k_orderb, size_t> seq2b(0);
	sequence<k_orderc, size_t> seq1(0), seq2(0);

	for(size_t i = 0; i < k_orderc; i++) seq1[i] = i;

	for(size_t i = 0; i < k_ordera; i++)
		seq2a[i] = label_c.index_of(label_a.letter_at(i));
	for(size_t i = 0; i < k_orderb; i++)
		seq2b[i] = label_c.index_of(label_b.letter_at(i));

	perma.apply(seq2a);
	permb.apply(seq2b);

	for(register size_t i = 0; i < k_ordera; i++)
		seq2[i] = seq2a[i];
	for(register size_t i = 0; i < k_orderb; i++)
		seq2[k_ordera + i] = seq2b[i];

	permutation_builder<k_orderc> permc(seq1, seq2);
	return permc.get_perm();
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_DIRSUM_PERMUTATION_BUILDER_H
