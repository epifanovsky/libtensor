#ifndef LIBTENSOR_SO_MULT_H
#define LIBTENSOR_SO_MULT_H

#include "../core/symmetry.h"
#include "symmetry_operation_base.h"
#include "symmetry_operation_params.h"

namespace libtensor {


template<size_t N, typename T>
class so_mult;

template<size_t N, typename T>
class symmetry_operation_params< so_mult<N, T> >;


/**	\brief Computes the %symmetry of the element-wise product of two tensors
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_mult : public symmetry_operation_base< so_mult<N, T> > {
private:
	typedef so_mult<N, T> operation_t;
	typedef symmetry_operation_dispatcher<operation_t> dispatcher_t;

private:
	const symmetry<N, T> &m_sym1; //!< First symmetry container (A)
	permutation<N> m_perm1; //!< Permutation of the first %tensor
	const symmetry<N, T> &m_sym2; //!< Second symmetry container (B)
	permutation<N> m_perm2; //!< Permutation of the second %tensor

public:
	/**	\brief Initializes the operation
		\param sym1 First %symmetry container (A).
		\param perm1 Permutation of the first %tensor.
		\param sym2 Second %symmetry container (B).
		\param perm2 Permutation of the second %tensor.
	 **/
	so_mult(const symmetry<N, T> &sym1, const permutation<N> &perm1,
		const symmetry<N, T> &sym2, const permutation<N> &perm2) :
		m_sym1(sym1), m_perm1(perm1), m_sym2(sym2), m_perm2(perm2) { }

	/**	\brief Performs the operation
		\param sym3 Destination %symmetry container.
	 **/
	void perform(symmetry<N, T> &sym3);

private:
	so_mult(const so_mult<N, T>&);
	const so_mult<N, T> &operator=(const so_mult<N, T>&);
};


template<size_t N, typename T>
void so_mult<N, T>::perform(symmetry<N, T> &sym3) {

	sym3.clear();

	for(typename symmetry<N, T>::iterator i1 = m_sym1.begin();
		i1 != m_sym1.end(); i1++) {

		const symmetry_element_set<N, T> &set1 =
			m_sym1.get_subset(i1);

		typename symmetry<N, T>::iterator i2 = m_sym2.begin();
		for(; i2 != m_sym2.end(); i2++) {
			if(m_sym2.get_subset(i2).get_id().
				compare(set1.get_id()) == 0) break;
		}

		if(i2 == m_sym2.end()) continue;

		const symmetry_element_set<N, T> &set2 =
			m_sym2.get_subset(i2);
		symmetry_element_set<N, T> set3(set1.get_id());
		symmetry_operation_params<operation_t> params(
			set1, m_perm1, set2, m_perm2, set3);
		dispatcher_t::get_instance().invoke(set1.get_id(), params);

		for(typename symmetry_element_set<N, T>::iterator j =
			set3.begin(); j != set3.end(); j++) {
			sym3.insert(set3.get_elem(j));
		}
	}
}


template<size_t N, typename T>
class symmetry_operation_params< so_mult<N, T> > :
	public symmetry_operation_params_i {

public:
	const symmetry_element_set<N, T> &grp1; //!< Symmetry group 1
	permutation<N> perm1; //!< Permutation 1
	const symmetry_element_set<N, T> &grp2; //!< Symmetry group 2
	permutation<N> perm2; //!< Permutation 2
	symmetry_element_set<N, T> &grp3; //!< Symmetry group 3 (output)

public:
	symmetry_operation_params(
		const symmetry_element_set<N, T> &grp1_,
		const permutation<N> &perm1_,
		const symmetry_element_set<N, T> &grp2_,
		const permutation<N> &perm2_,
		symmetry_element_set<N, T> &grp3_) :

		grp1(grp1_), perm1(perm1_), grp2(grp2_), perm2(perm2_),
		grp3(grp3_) { }

	virtual ~symmetry_operation_params() { }
};


} // namespace libtensor

#include "so_mult_handlers.h"

#endif // LIBTENSOR_SO_MULT_H
