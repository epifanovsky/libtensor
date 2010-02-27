#ifndef LIBTENSOR_PERMUTATION_GROUP_H
#define LIBTENSOR_PERMUTATION_GROUP_H

#include <list>
#include "../defs.h"
#include "symmetry_element_set_adapter.h"
#include "se_perm.h"

namespace libtensor {


/**	Manipulates a %permutation group
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The %permutation group is a subgroup of the symmetric group. This
	class reconstructs a particular subgroup using the generating set
	and provides a set of operations on the subgroup.

	The %permutation group is represented as a direct product of disjoint
	(not acting on the same %tensor indexes) subgroups:
	\f[ P = P^{(1)} \otimes P^{(2)} \otimes \cdots \otimes P^{(n)},
		n \leq N \f]
	If there is no %symmetry, \f$ P^{(i)} = C_1, 1 \leq i \leq N \f$.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class permutation_group {
public:
	static const char *k_clazz; //!< Class name

private:
	//!	Represents one of the disjoint subgroups
	struct subgroup {
		mask<N> msk; //!< Mask of affected indexes
		sequence<N, size_t> cycles; //!< Map of cycles
		bool sign; //!< Symmetric (T), anti-symmetric (F)
		bool sym; //!< Symmetric (T), cyclic (F)

		subgroup() : cycles(0), sign(false), sym(false) { }

		subgroup(const mask<N> &msk_,
			const sequence<N, size_t> &cycles_, bool sign_,
			bool sym_) : msk(msk_), cycles(cycles_), sign(sign_),
			sym(sym_) { }
	};

private:
	typedef se_perm<N, T> se_perm_t;
	typedef std::list<subgroup> subgroup_list_t;

private:
	subgroup_list_t m_list; //!< List of disjoint subgroups

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the C1 group
	 **/
	permutation_group() { }

	/**	\brief Creates a %permutation group from a generating set
	 **/
	permutation_group(
		const symmetry_element_set_adapter<N, T, se_perm_t> &set);

	/**	\brief Destroys the object
	 **/
	~permutation_group() { }

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Joins a %permutation with the group
		\param perm Permutation.
		\param sign Symmetric (true), anti-symmetric (false).
	 **/
	void join_permutation(const permutation<N> &perm, bool sign);

	/**	\brief Converts the %permutation group to a generating set
	 **/
	void convert(symmetry_element_set<N, T> &set);

	//@}

private:
	/**	\brief Makes the %mask of affected indexes and a cycle map
			of a %permutation. Returns the number of cycles.
	 **/
	size_t mask_and_map(const permutation<N> &perm, mask<N> &msk,
		sequence<N, size_t> &cycles);
};


template<size_t N, typename T>
const char *permutation_group<N, T>::k_clazz = "permutation_group<N, T>";


template<size_t N, typename T>
permutation_group<N, T>::permutation_group(
	const symmetry_element_set_adapter<N, T, se_perm_t> &set) {

	typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;
	for(typename adapter_t::iterator i = set.begin(); i != set.end(); i++) {
		const se_perm_t &e = set.get_elem(i);
		join_permutation(e.get_perm(), e.is_symm());
	}
}


template<size_t N, typename T>
void permutation_group<N, T>::join_permutation(
	const permutation<N> &perm, bool sign) {

	mask<N> msk1;
	sequence<N, size_t> cyc1;
	size_t ncyc1 = mask_and_map(perm, msk1, cyc1);
}


template<size_t N, typename T>
void permutation_group<N, T>::convert(symmetry_element_set<N, T> &set) {

}


template<size_t N, typename T>
size_t permutation_group<N,  T>::mask_and_map(
	const permutation<N> &perm, mask<N> &msk, sequence<N, size_t> &cycles) {

	static const char *method = "mask_and_map(const permutation<N>&, "
		"mask<N>&, sequence<N, size_t>&)";

	//	Analyze the permutation:
	//	 * Mark indexes that are affected by the permutation
	//	 * Enumerate cycles and create a map of cycles
	//
	//	In the map of cycles, each cycle is numbered starting from 1.
	//	Unaffected indexes are marked 0.
	//
	//	For example, [012345->120435] has five affected indexes and
	//	two cycles. One index stays in place after the permutation.
	//	The result will be: msk = [111110], cycles = [111220]

	size_t seq[N];
	for(size_t i = 0; i < N; i++) seq[i] = i;
	perm.apply(seq);
	mask<N> visited;
	size_t ncycles = 0;
	for(register size_t ic = 0; ic < N; ic++) {

		if(visited[ic]) continue;

		register size_t ic1 = seq[ic];
		if(ic1 == ic) {
			visited[ic] = true;
			continue;
		}

		ncycles++;
		cycles[ic] = ncycles;
		visited[ic] = true;
		msk[ic] = true;
		while(ic1 != ic) {
			cycles[ic1] = ncycles;
			visited[ic1] = true;
			msk[ic1] = true;
			ic1 = seq[ic1];
		}
	}

	return ncycles;
}


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_H