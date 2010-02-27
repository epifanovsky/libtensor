#ifndef LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H
#define LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "so_proj_down.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_proj_down<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_proj_down_impl< se_perm<N, T> > {
public:
	static const char *k_clazz; //!< Class name

private:
	struct subgroup {
		permutation<N> perm;
		mask<N> msk;
		sequence<N, size_t> cycles;
		bool sign; // T=symmetric/F=antisymmetric
		bool sym; // cyclic/symmetric
		subgroup() : cycles(0), sign(false), sym(false) { }
		subgroup(const permutation<N> &perm_, const mask<N> &msk_,
			const sequence<N, size_t> &cycles_, bool sign_,
			bool sym_) : perm(perm_), msk(msk_), cycles(cycles_),
			sign(sign_), sym(sym_) { }
	};

public:
	template<size_t M>
	void perform(
		const symmetry_operation_params<
			so_proj_down<N, M, T> > &params,
		symmetry_element_set<N - M, T> &set);

private:
	/**	\brief Makes the %mask of affected indexes and a cycle map
			of a %permutation
	 **/
	void mask_and_map(const permutation<N> &perm, mask<N> &msk,
		sequence<N, size_t> &cycles);

	/**	\brief Expands the group description by adding a generating
			element
	 **/
	void join_group(std::list<subgroup> &grp, const permutation<N> &perm,
		mask<N> &msk, sequence<N, size_t> &cycles, bool sign);
};


template<size_t N, typename T>
const char *so_proj_down_impl< se_perm<N, T> >::k_clazz =
	"so_proj_down_impl< se_perm<N, T> >";


template<size_t N, typename T> template<size_t M>
void so_proj_down_impl< se_perm<N, T> >::perform(
	const symmetry_operation_params< so_proj_down<N, M, T> > &params,
	symmetry_element_set<N - M, T> &set) {

	static const char *method =
		"perform<M>(const symmetry_operation_params< "
		"so_proj_down<N, M, T> >&, symmetry_element_set<N - M, T>&)";

	//	Adapter type for the input group
	//
	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> >
		adapter_t;

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;
	if(nm != N - M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	//	Reconstruct the symmetry group
	//
	adapter_t g1(params.grp);
	std::list<subgroup> group;
	for(typename adapter_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_perm<N, T> &e1 = g1.get_elem(i);

		//	Compute the mask of the permutation
		//
		sequence<N, size_t> cycles(0);
		mask<N> msk;
		mask_and_map(e1.get_perm(), msk, cycles);

		//	Insert the current element into the group
		//
		join_group(group, e1.get_perm(), msk, cycles, e1.is_symm());
	}

	//	Go over subgroups and project them
	//
	mask<N> m0;
	for(typename std::list<subgroup>::iterator i = group.begin();
		i != group.end(); i++) {

		mask<N> m1 = i->msk & params.msk;

		//	Reject if the subgroup is projected out entirely
		//
		if(m1.equals(m0)) continue;

		//	Accept if the subgroup is entirely in the subspace
		//
		if(m1.equals(i->msk)) {
			const mask<N> &msk = i->msk;
			size_t seq1a[N], seq1b[N];
			size_t seq2a[N - M], seq2b[N - M];
			for(size_t j = 0; j < N; j++) seq1b[j] = seq1a[j] = j;
			i->perm.apply(seq1b);
			for(size_t j = 0, k = 0; j < N; j++) {
				if(msk[j]) {
					seq2a[k] = seq1a[j];
					seq2b[k] = seq1b[j];
					k++;
				}
			}
			permutation_builder<N - M> pb(seq2b, seq2a);
			set.insert(se_perm<N - M, T>(pb.get_perm(), i->sign));
			continue;
		}

		//	Partial overlap: reject cyclic subgroups
		//
		if(!i->sym) continue;

		//	Partial overlap: project symmetric subgroups
		//
		
		throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
		
	}
}


template<size_t N, typename T>
void so_proj_down_impl< se_perm<N, T> >::mask_and_map(
	const permutation<N> &perm, mask<N> &msk, sequence<N, size_t> &cycles) {

	static const char *method = "mask_and_map(const permutation<N>&, "
		"mask<N>&, sequence<N, size_t>&)";

	size_t seq[N];
	for(size_t i = 0; i < N; i++) seq[i] = i;
	perm.apply(seq);
	mask<N> visited;
	size_t ncycles = 0;
	for(size_t ic = 0; ic < N; ic++) {

		if(visited[ic]) continue;

		size_t ic1 = seq[ic];
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
}


template<size_t N, typename T>
void so_proj_down_impl< se_perm<N, T> >::join_group(std::list<subgroup> &grp,
	const permutation<N> &perm, mask<N> &msk, sequence<N, size_t> &cycles,
	bool sign) {

	static const char *method = "join_group(std::list<subgroup>&, "
		"const permutation<N>&, mask<N>&, sequence<N, size_t>&, bool)";

	mask<N> m0;
	bool joined = false;
	for(typename std::list<subgroup>::iterator i = grp.begin();
		i != grp.end(); i++) {

		//	Ignore those generating elements that don't overlap
		//
		if((i->msk & msk).equals(m0)) continue;

		//	Compute cycle lengths
		//
		size_t clen1[N], clen2[N];
		size_t nc1 = 0, nc2 = 0, maxclen1 = 0, maxclen2 = 0;
		for(size_t j = 0; j < N; j++) clen2[j] = clen1[j] = 0;
		for(size_t j = 0; j < N; j++) {
			register size_t c1 = cycles[j], c2 = i->cycles[j];
			if(c1 > nc1) nc1 = c1;
			clen1[c1]++;
			if(clen1[c1] > maxclen1) maxclen1 = clen1[c1];
			if(c2 > nc2) nc2 = c2;
			clen2[c2]++;
			if(clen2[c2] > maxclen2) maxclen2 = clen2[c2];
		}

		//	Make sure that 0-cycles correspond to
		//	0-cycles or 2-cycles
		//

		//	Make sure that n-cycles (n > 2) correspond to 2-cycles
		//
		throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
	}

	//	If there's no overlap with existing subgroups,
	//	simply add on the list
	//
	if(!joined) {
		subgroup sgrp(perm, msk, cycles, sign, false);
		grp.push_back(sgrp);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H