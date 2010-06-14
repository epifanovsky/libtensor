#ifndef LIBTENSOR_SO_INTERSECTION_H
#define LIBTENSOR_SO_INTERSECTION_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_intersection;

template<typename ElemT>
class so_intersection_impl;

template<size_t N, typename T>
class symmetry_operation_params< so_intersection<N, T> >;


/**	\brief Intersection of two %symmetry groups

	The intersection of two %symmetry groups \f$ G = G_A \cap G_B \f$
	combines the elements present in both groups such that the result
	is also a group.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_intersection :
	public symmetry_operation_base< so_intersection<N, T> > {

private:
	typedef so_intersection<N, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
	const symmetry<N, T> &m_sym1;
	const symmetry<N, T> &m_sym2;

public:
	/**	\brief Initializes the operation
		\param sym1 First %symmetry group.
		\param sym2 Second %symmetry group.
	 **/
	so_intersection(const symmetry<N, T> &sym1, const symmetry<N, T> &sym2) :
		m_sym1(sym1), m_sym2(sym2) { }

	/**	\brief Performs the operation
		\param sym3 Output %symmetry group.
	 **/
	void perform(symmetry<N, T> &sym3);

private:
	void copy_subset(const symmetry_element_set<N, T> &set,
			symmetry<N, T> &sym2);
};

template<size_t N, typename T>
void so_intersection<N, T>::perform(symmetry<N, T> &sym3) {

	sym3.clear();

	for(typename symmetry<N, T>::iterator i = m_sym1.begin();
			i != m_sym1.end(); i++) {

		const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

		for (typename symmetry<N, T>::iterator j = m_sym2.begin();
				j != m_sym2.end(); j++) {

			const symmetry_element_set<N, T> &set2 = m_sym2.get_subset(j);

			if (set1.get_id() == set2.get_id()) {
				symmetry_element_set<N, T> set3(set1.get_id());
				symmetry_operation_params<operation_t> params(set1, set2, set3);
				dispatcher_t::get_instance().invoke(set1.get_id(), params);
				copy_subset(set3, sym3);
			}
		}
	}
}

template<size_t N, typename T>
void so_intersection<N, T>::copy_subset(const symmetry_element_set<N, T> &set,
		symmetry<N, T> &sym) {

	for (typename symmetry_element_set<N, T>::const_iterator i = set.begin();
			i != set.end(); i++) {

		sym.insert(set.get_elem(i));
	}
}

template<size_t N, typename T>
class symmetry_operation_params< so_intersection<N, T> > :
	public symmetry_operation_params_i {
public:
	const symmetry_element_set<N, T> &g1;
	const symmetry_element_set<N, T> &g2;
	symmetry_element_set<N, T> &g3;

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<N, T> &g2_,
		symmetry_element_set<N, T> &g3_) : g1(g1_), g2(g2_), g3(g3_) { }

	virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_intersection_handlers.h"


#endif // LIBTENSOR_SO_INTERSECTION_H

