#ifndef LIBTENSOR_SO_CONCAT_H
#define LIBTENSOR_SO_CONCAT_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "so_permute.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, size_t M, typename T>
class so_concat;

template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_concat<N, M, T> >;


/**	\brief Concatenates two %symmetry groups to form a larger space
	\tparam N Order of the argument space.
	\tparam M Increment in the order of the result space.

	The operation forms the direct product of two given %symmetry groups
	resulting in a larger space.

	The operation takes two %symmetry group that are defined for %tensor
	spaces of order N and M, respectively and produces a group that acts in
	a %tensor space of order N + M.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class so_concat : public symmetry_operation_base< so_concat<N, M, T> > {
private:
	typedef so_concat<N, M, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
	const symmetry<N, T> &m_sym1;
	const symmetry<M, T> &m_sym2;
	permutation<N + M> m_perm;

public:
	so_concat(const symmetry<N, T> &sym1, const symmetry<M, T> &sym2,
			const permutation<N + M> &perm) :
				m_sym1(sym1), m_sym2(sym2), m_perm(perm) { }

	so_concat(const symmetry<N, T> &sym1, const symmetry<M, T> &sym2) :
		m_sym1(sym1), m_sym2(sym2) { }

	void perform(symmetry<N + M, T> &sym3);

};


/**	\brief Concatenate vacuum with other symmetry (specialization)
	\tparam M Order.

	\ingroup libtensor_symmetry
 **/
template<size_t M, typename T>
class so_concat<0, M, T> {
private:
	const symmetry<M, T> &m_sym2;
	permutation<M> m_perm;

public:
	so_concat(const symmetry<0, T> &sym1, const symmetry<M, T> &sym2) :
		m_sym2(sym2) { }
	so_concat(const symmetry<0, T> &sym1, const symmetry<M, T> &sym2,
			const permutation<M> &perm) : m_sym2(sym2), m_perm(perm) { }

	void perform(symmetry<M, T> &sym3) {

		sym3.clear();
		so_permute<M, T>(m_sym2, m_perm).perform(sym3);
	}
};

/**	\brief Concatenate symmetry with vacuum (specialization)
	\tparam N Order.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_concat<N, 0, T> {
private:
	const symmetry<N, T> &m_sym1;
	permutation<N> m_perm;

public:
	so_concat(const symmetry<N, T> &sym1, const symmetry<0, T> &sym2) :
		m_sym1(sym1) { }
	so_concat(const symmetry<N, T> &sym1, const symmetry<0, T> &sym2,
			const permutation<N> &perm) : m_sym1(sym1), m_perm(perm) { }

	void perform(symmetry<N, T> &sym3) {

		sym3.clear();
		so_permute<N, T>(m_sym1, m_perm).perform(sym3);
	}
};


template<size_t N, size_t M, typename T>
void so_concat<N, M, T>::perform(symmetry<N + M, T> &sym3) {

	for(typename symmetry<N, T>::iterator i1 = m_sym1.begin();
		i1 != m_sym1.end(); i1++) {

		const symmetry_element_set<N, T> &set1 =
			m_sym1.get_subset(i1);

		typename symmetry<M, T>::iterator i2 = m_sym2.begin();
		for(; i2 != m_sym2.end(); i2++) {
			if(m_sym2.get_subset(i2).get_id().
				compare(set1.get_id()) == 0) break;
		}

		if(i2 == m_sym2.end()) continue;

		const symmetry_element_set<M, T> &set2 =
			m_sym2.get_subset(i2);
		symmetry_element_set<N + M, T> set3(set1.get_id());
		symmetry_operation_params<operation_t> params(
			set1, set2, m_perm, set3);
		dispatcher_t::get_instance().invoke(set1.get_id(), params);

		for(typename symmetry_element_set<N + M, T>::iterator j =
			set3.begin(); j != set3.end(); j++) {
			sym3.insert(set3.get_elem(j));
		}
	}
}


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_concat<N, M, T> > :
	public symmetry_operation_params_i {

public:
	const symmetry_element_set<N, T> &g1; //!< Symmetry group
	const symmetry_element_set<M, T> &g2; //!< Symmetry group
	permutation<N + M> perm; //!< Permutation
	symmetry_element_set<N + M, T> &g3;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<M, T> &g2_,
		const permutation<N + M> &perm_,
		symmetry_element_set<N + M, T> &g3_) :

		g1(g1_), g2(g2_), perm(perm_), g3(g3_)  { }

};


} // namespace libtensor

#include "so_concat_handlers.h"

#endif // LIBTENSOR_SO_CONCAT_H
