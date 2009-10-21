#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CONTRACTION2_BUILDER_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CONTRACTION2_BUILDER_H

#include "defs.h"
#include "exception.h"
#include "core/permutation_builder.h"
#include "tod/contraction2.h"
#include "iface/letter_expr.h"

namespace libtensor {
namespace labeled_btensor_expr {


/**	\brief Builds the contraction2 object using labels
	\tparam N Order of the first %tensor (A) less contraction degree.
	\tparam M Order of the second %tensor (B) less contraction degree.
	\tparam K Number of indexes contracted.

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, size_t K>
class contract_contraction2_builder {
public:
	static const size_t k_ordera = N + K; //!< Order of the first %tensor
	static const size_t k_orderb = M + K; //!< Order of the second %tensor
	static const size_t k_orderc = N + M; //!< Order of the result

private:
	contraction2<N, M, K> m_contr;

public:
	contract_contraction2_builder(
		const letter_expr<k_ordera> &label_a,
		const permutation<k_ordera> &perm_a,
		const letter_expr<k_orderb> &label_b,
		const permutation<k_orderb> &perm_b,
		const letter_expr<k_orderc> &label_c,
		const letter_expr<K> &contr);

	const contraction2<N, M, K> &get_contr() const { return m_contr; }

private:
	static contraction2<N, M, K> mk_contr(
		const letter_expr<k_ordera> &label_a,
		const letter_expr<k_orderb> &label_b,
		const letter_expr<k_orderc> &label_c,
		const letter_expr<K> &contr);
};


template<size_t N, size_t M, size_t K>
contract_contraction2_builder<N, M, K>::contract_contraction2_builder(
	const letter_expr<k_ordera> &label_a,
	const permutation<k_ordera> &perm_a,
	const letter_expr<k_orderb> &label_b,
	const permutation<k_orderb> &perm_b,
	const letter_expr<k_orderc> &label_c, const letter_expr<K> &contr) :

	m_contr(mk_contr(label_a, label_b, label_c, contr)) {

	m_contr.permute_a(perm_a);
	m_contr.permute_b(perm_b);

}


template<size_t N, size_t M, size_t K>
contraction2<N, M, K> contract_contraction2_builder<N, M, K>::mk_contr(
	const letter_expr<k_ordera> &label_a,
	const letter_expr<k_orderb> &label_b,
	const letter_expr<k_orderc> &label_c, const letter_expr<K> &contr) {

	size_t contr_a[K], contr_b[K];
	size_t seq1[k_orderc], seq2[k_orderc];

	for(size_t i = 0; i < k_orderc; i++) seq1[i] = i;

	size_t j = 0, k = 0;
	for(size_t i = 0; i < k_ordera; i++) {
		const letter &l = label_a.letter_at(i);
		if(label_c.contains(l)) {
			seq2[j] = label_c.index_of(l);
			j++;
		} else {
			if(!contr.contains(l)) {
				throw_exc("contract_contraction2_builder<N, M, K>", "mk_contr()",
					"Inconsistent expression.");
			}
			contr_a[k] = i;
			contr_b[k] = label_b.index_of(l);
			k++;
		}
	}
	for(size_t i = 0; i < k_orderb; i++) {
		const letter &l = label_b.letter_at(i);
		if(label_c.contains(l)) {
			seq2[j] = label_c.index_of(l);
			j++;
		}
	}

	permutation_builder<k_orderc> permc(seq1, seq2);
	std::cout << "permc " << permc.get_perm() << std::endl;
	contraction2<N, M, K> c(permc.get_perm());

	for(size_t i = 0; i < K; i++) {
		c.contract(contr_a[i], contr_b[i]);
	}

	return c;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CONTRACT_CONTRACTION2_BUILDER_H
