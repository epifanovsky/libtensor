#ifndef LIBTENSOR_SO_UNION_H
#define LIBTENSOR_SO_UNION_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_dispatcher.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_union;

template<size_t N, typename T>
class symmetry_operation_params< so_union<N, T> >;


/**	\brief Union of two %symmetry groups

	The union of two %symmetry groups \f$ G = G_A \cup G_B \f$ combines
	the elements of the groups such that the result is also a group.
	If the union is not a group, the operation causes an exception.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_union {
private:
	typedef so_union<N, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
	const symmetry<N, T> &m_sym1;
	const symmetry<N, T> &m_sym2;

public:
	/**	\brief Initializes the operation
		\param sym1 First %symmetry group.
		\param sym2 Second %symmetry group.
	 **/
	so_union(const symmetry<N, T> &sym1, const symmetry<N, T> &sym2) :
		m_sym1(sym1), m_sym2(sym2) { }

	/**	\brief Performs the operation
		\param sym3 Output %symmetry group.
	 **/
	void perform(symmetry<N, T> &sym3);

private:
	void do_subset_pass1(const symmetry_element_set<N, T> &set1,
		const symmetry<N, T> &sym2, symmetry<N, T> &sym3);
	void do_subset_pass2(const symmetry_element_set<N, T> &set1,
		const symmetry<N, T> &sym2, symmetry<N, T> &sym3);
	void copy_subset(const symmetry_element_set<N, T> &set1,
		symmetry<N, T> &sym2);

};


template<size_t N, typename T>
void so_union<N, T>::perform(symmetry<N, T> &sym3) {

	sym3.clear();

	for(typename symmetry<N, T>::iterator i = m_sym1.begin();
		i != m_sym1.end(); i++) {

		do_subset_pass1(m_sym1.get_subset(i), m_sym2, sym3);
	}
	for(typename symmetry<N, T>::iterator i = m_sym2.begin();
		i != m_sym2.end(); i++) {

		do_subset_pass2(m_sym2.get_subset(i), m_sym1, sym3);
	}
}


template<size_t N, typename T>
void so_union<N, T>::do_subset_pass1(const symmetry_element_set<N, T> &set1,
	const symmetry<N, T> &sym2, symmetry<N, T> &sym3) {

	typename symmetry<N, T>::iterator i;
	for(i = sym2.begin(); i != sym2.end(); i++) {
		if(set1.get_id() == sym2.get_subset(i).get_id()) break;
	}

	if(i == sym2.end()) {
		copy_subset(set1, sym3);
	} else {
		symmetry_element_set<N, T> set3(set1.get_id());
		const symmetry_element_set<N, T> &set2 = sym2.get_subset(i);
		symmetry_operation_params<operation_t> params(set1, set2, set3);
		dispatcher_t::get_instance().invoke(set1.get_id(), params);
		copy_subset(set3, sym3);
	}
}


template<size_t N, typename T>
void so_union<N, T>::do_subset_pass2(const symmetry_element_set<N, T> &set1,
	const symmetry<N, T> &sym2, symmetry<N, T> &sym3) {

	for(typename symmetry<N, T>::iterator i = sym2.begin();
		i != sym2.end(); i++) {

		if(set1.get_id() == sym2.get_subset(i).get_id()) return;
	}

	copy_subset(set1, sym3);
}


template<size_t N, typename T>
void so_union<N, T>::copy_subset(const symmetry_element_set<N, T> &set1,
	symmetry<N, T> &sym2) {

	for(typename symmetry_element_set<N, T>::const_iterator i =
		set1.begin(); i != set1.end(); i++) {

		sym2.insert(set1.get_elem(i));
	}
}


template<size_t N, typename T>
class symmetry_operation_params< so_union<N, T> > :
	public symmetry_operation_params_i {

public:
	const symmetry_element_set<N, T> &g1;
	const symmetry_element_set<N, T> &g2;
	symmetry_element_set<N, T> &g3;

public:
	symmetry_operation_params(const symmetry_element_set<N, T> &g1_,
		const symmetry_element_set<N, T> &g2_,
		symmetry_element_set<N, T> &g3_) : g1(g1_), g2(g2_), g3(g3_) { }

	virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_union_handlers.h"

#endif // LIBTENSOR_SO_UNION_H
