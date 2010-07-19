#ifndef LIBTENSOR_SO_PROJ_UP_H
#define LIBTENSOR_SO_PROJ_UP_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_proj_up;

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_proj_up<N, M, T> >;


/**	\brief Projection of a %symmetry group to a larger space
	\tparam N Order of the argument space.
	\tparam M Increment in the order of the result space.

	The operation projects a given %symmetry group to a larger space.
	The resulting group will affect only the subspace of the large space
	specified by a %mask.

	The operation takes a %symmetry group that is defined for a %tensor
	space of order N and produces a group that acts in a %tensor space of
	order N + M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_proj_up : public symmetry_operation_base< so_proj_up<N, M, T> > {
private:
	typedef so_proj_up<N, M, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
	const symmetry<N, T> &m_sym1;
	permutation<N> m_perm;
	mask<N + M> m_msk;

public:
	so_proj_up(const symmetry<N, T> &sym1, const permutation<N> &perm,
		const mask<N + M> &msk) : m_sym1(sym1), m_perm(perm), m_msk(msk)
		{ }

	so_proj_up(const symmetry<N, T> &sym1, const mask<N + M> &msk) :
		m_sym1(sym1), m_msk(msk) { }

	void perform(symmetry<N + M, T> &sym2);

};


/**	\brief Projection of vacuum onto a larger space (specialization)
	\tparam M Order.

	\ingroup libtensor_symmetry
 **/
template<size_t M, typename T>
class so_proj_up<0, M, T> {
public:
	so_proj_up(const symmetry<0, T> &sym1, const mask<M> &msk) { }
	so_proj_up(const symmetry<0, T> &sym1, const permutation<0> &perm,
		const mask<M> &msk) { }

	void perform(symmetry<M, T> &sym2) {
		sym2.clear();
	}
};


template<size_t N, size_t M, typename T>
void so_proj_up<N, M, T>::perform(symmetry<N + M, T> &sym2) {

	for(typename symmetry<N, T>::iterator i = m_sym1.begin();
		i != m_sym1.end(); i++) {

		const symmetry_element_set<N, T> &set1 =
			m_sym1.get_subset(i);
		symmetry_element_set<N + M, T> set2(set1.get_id());
		symmetry_operation_params<operation_t> params(
			set1, m_perm, m_msk, sym2.get_bis(), set2);
		dispatcher_t::get_instance().invoke(set1.get_id(), params);

		for(typename symmetry_element_set<N + M, T>::iterator j =
			set2.begin(); j != set2.end(); j++) {
			sym2.insert(set2.get_elem(j));
		}
	}
}


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_proj_up<N, M, T> > :
	public symmetry_operation_params_i {

public:
	const symmetry_element_set<N, T> &g1; //!< Symmetry group
	permutation<N> perm; //!< Permutation
	mask<N + M> msk; //!< Mask
	const block_index_space<N + M> &bis; //!< Block index space of result
	symmetry_element_set<N + M, T> &g2;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const permutation<N> &perm_, const mask<N + M> &msk_,
		const block_index_space<N + M> &bis_,
		symmetry_element_set<N + M, T> &g2_) :

		g1(g1_), perm(perm_), msk(msk_), bis(bis_), g2(g2_) { }

};


} // namespace libtensor

#include "so_proj_up_handlers.h"

#endif // LIBTENSOR_SO_PROJ_UP_H
