#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Builds permutations for the element-wise product

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K>
class ewmult_perm_builder {
public:
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M + K; //!< Order of the result

private:
	permutation<k_ordera> m_perma;
	permutation<k_orderb> m_permb;
	permutation<k_orderc> m_permc;

public:
	ewmult_perm_builder(
		const letter_expr<k_ordera> &label_a,
		const letter_expr<k_orderb> &label_b,
		const letter_expr<k_orderc> &label_c,
		const letter_expr<K> &ewidx);

	const permutation<k_ordera> &get_perma() const {
		return m_perma;
	}

	const permutation<k_orderb> &get_permb() const {
		return m_permb;
	}

	const permutation<k_orderc> &get_permc() const {
		return m_permc;
	}

};


template<size_t N, size_t M, size_t K>
ewmult_perm_builder<N, M, K>::ewmult_perm_builder(
	const letter_expr<k_ordera> &label_a,
	const letter_expr<k_orderb> &label_b,
	const letter_expr<k_orderc> &label_c, const letter_expr<K> &ewidx) {

	sequence<k_ordera, const letter*> seqa1(0), seqa2(0);
	sequence<k_orderb, const letter*> seqb1(0), seqb2(0);
	sequence<k_orderc, const letter*> seqc1(0), seqc2(0);

	size_t k = 0;
	for(size_t i = 0, j = 0; i < k_ordera; i++) {
		const letter &l = label_a.letter_at(i);
		seqa1[i] = &l;
		if(!ewidx.contains(l)) {
			seqc1[k++] = seqa2[j++] = &l;
		}
	}

	for(size_t i = 0, j = 0; i < k_orderb; i++) {
		const letter &l = label_b.letter_at(i);
		seqb1[i] = &l;
		if(!ewidx.contains(l)) {
			seqc1[k++] = seqb2[j++] = &l;
		}
	}
	for(size_t i = 0; i < K; i++) {
		seqc1[N + M + i] = seqa2[N + i] = seqb2[M + i] =
			&ewidx.letter_at(i);
	}

	for(size_t i = 0; i < k_orderc; i++) seqc2[i] = &label_c.letter_at(i);

	m_perma.permute(permutation_builder<k_ordera>(seqa2, seqa1).get_perm());
	m_permb.permute(permutation_builder<k_orderb>(seqb2, seqb1).get_perm());
	m_permc.permute(permutation_builder<k_orderc>(seqc2, seqc1).get_perm());
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EWMULT_PERM_BUILDER_H
