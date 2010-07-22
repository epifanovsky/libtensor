#ifndef LIBTENSOR_SO_CONCAT_H
#define LIBTENSOR_SO_CONCAT_H

#include "../core/mask.h"
#include "../core/symmetry.h"
#include "../core/symmetry_element_set.h"
#include "so_permute.h"
#include "so_proj_up.h"
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

private:
	template<size_t X>
	void copy_subset(const symmetry_element_set<X, T> &set1,
		symmetry<X, T> &sym2);
	void proj_up_sym1(const symmetry<N, T> &sym1, symmetry<N + M, T> &sym2);
	void proj_up_sym2(const symmetry<M, T> &sym1, symmetry<N + M, T> &sym2);

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

	sym3.clear();

	symmetry<N, T> sym1(m_sym1.get_bis());

	for(typename symmetry<N, T>::iterator i = m_sym1.begin();
		i != m_sym1.end(); i++) {

		const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

		typename symmetry<M, T>::iterator j;
		for(j = m_sym2.begin(); j != m_sym2.end(); j++) {
			if(set1.get_id() == m_sym2.get_subset(j).get_id()) break;
		}

		if(j == m_sym2.end()) {
			copy_subset(set1, sym1);
		} else {
			symmetry_element_set<N + M, T> set3(set1.get_id());
			const symmetry_element_set<M, T> &set2 = m_sym2.get_subset(j);
			symmetry_operation_params<operation_t> params(
					set1, set2, m_perm, sym3.get_bis(), set3);
			dispatcher_t::get_instance().invoke(set1.get_id(), params);
			copy_subset(set3, sym3);
		}
	}

	if (sym1.begin() != sym1.end())
		proj_up_sym1(sym1, sym3);

	symmetry<M, T> sym2(m_sym2.get_bis());

	for(typename symmetry<M, T>::iterator i = m_sym2.begin();
		i != m_sym2.end(); i++) {

		const symmetry_element_set<M, T> &set2 = m_sym2.get_subset(i);

		typename symmetry<N, T>::iterator j;
		for(j = m_sym1.begin(); j != m_sym1.end(); j++) {

			if(set2.get_id() == m_sym1.get_subset(j).get_id()) break;
		}

		if (j == m_sym1.end())
			copy_subset(set2, sym2);

	}

	if (sym2.begin() != sym2.end())
		proj_up_sym2(sym2, sym3);
}




template<size_t N, size_t M, typename T>
template<size_t X>
void so_concat<N, M, T>::copy_subset(const symmetry_element_set<X, T> &set1,
	symmetry<X, T> &sym2) {

	for(typename symmetry_element_set<X, T>::const_iterator i =
		set1.begin(); i != set1.end(); i++) {

		sym2.insert(set1.get_elem(i));
	}
}

template<size_t N, size_t M, typename T>
void so_concat<N, M, T>::proj_up_sym1(const symmetry<N, T> &sym1,
	symmetry<N + M, T> &sym2) {

	size_t map[N + M];
	for (size_t i = 0; i < N + M; i++) map[i] = i;
	m_perm.apply(map);

	mask<N + M> msk;
	size_t a1[N], a2[N], j = 0;
	for (size_t i = 0; i < N + M; i++) {
		if (map[i] < N) {
			a1[j] = j;
			a2[j] = map[i];
			msk[i] = true;
			j++;
		}
	}
	permutation_builder<N> pb(a2, a1);

	so_proj_up<N, M, T>(sym1, pb.get_perm(), msk).perform(sym2);
}

template<size_t N, size_t M, typename T>
void so_concat<N, M, T>::proj_up_sym2(const symmetry<M, T> &sym1,
	symmetry<N + M, T> &sym2) {

	size_t map[N + M];
	for (size_t i = 0; i < N + M; i++) map[i] = i;
	m_perm.apply(map);

	mask<N + M> msk;
	size_t a1[M], a2[M], j = 0;
	for (size_t i = 0; i < N + M; i++) {
		if (map[i] >= N) {
			a1[j] = j;
			a2[j] = map[i] - N;
			msk[i] = true;
			j++;
		}
	}
	permutation_builder<M> pb(a2, a1);

	so_proj_up<M, N, T>(sym1, pb.get_perm(), msk).perform(sym2);
}


template<size_t N, size_t M, typename T>
class symmetry_operation_params< so_concat<N, M, T> > :
	public symmetry_operation_params_i {

public:
	const symmetry_element_set<N, T> &g1; //!< Symmetry group
	const symmetry_element_set<M, T> &g2; //!< Symmetry group
	permutation<N + M> perm; //!< Permutation
	block_index_space<N + M> bis; //!< Block index space of result
	symmetry_element_set<N + M, T> &g3;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<M, T> &g2_,
		const permutation<N + M> &perm_,
		const block_index_space<N + M> &bis_,
		symmetry_element_set<N + M, T> &g3_) :

		g1(g1_), g2(g2_), perm(perm_), bis(bis_), g3(g3_)  { }

};


} // namespace libtensor

#include "so_concat_handlers.h"

#endif // LIBTENSOR_SO_CONCAT_H
