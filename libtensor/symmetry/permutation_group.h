#ifndef LIBTENSOR_PERMUTATION_GROUP_H
#define LIBTENSOR_PERMUTATION_GROUP_H

#include <list>
#include "../defs.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "se_perm.h"

namespace libtensor {


/**	Stores the reduced representation of a permutation group
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This class implements the %permutation group container and procedures
	described in M. Jerrum, J. Algorithms 7 (1986), 60-78.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class permutation_group {
public:
	static const char *k_clazz; //!< Class name

private:
	//!	Stores one labeled branching
	struct branching {
		permutation<N> m_labels[N]; //!< Edge labels
		size_t m_edges[N]; //!< Edge sources
		branching() {
			for(register size_t i = 0; i < N; i++) m_edges[i] = N;
		}
	};

private:
	typedef se_perm<N, T> se_perm_t;

	typedef permutation<N> permutation_t;
	typedef std::list<permutation_t> perm_list_t;
	typedef std::vector<permutation_t> perm_vec_t;

private:
	branching m_symm; //!< Branching of the symmetric part
	branching m_asymm; //!< Branching of the anti-symmetric part

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the C1 group
	 **/
	permutation_group() { }

	/**	\brief Creates a %permutation group from a generating set
		\param set Generating set.
	 **/
	permutation_group(
		const symmetry_element_set_adapter<N, T, se_perm_t> &set);

	/**	\brief Destroys the object
	 **/
	~permutation_group() { }

	//@}


	//!	\name Manipulations
	//@{

	/**	\brief Tests the membership of a %permutation in the group
		\param perm Permutation.
		\param sign Symmetric (true)/anti-symmetric (false)
	 **/
	bool is_member(const permutation<N> &perm, bool sign);

	/**	\brief Converts the %permutation group to a generating set
			using the standard format
	 **/
	void convert(symmetry_element_set<N, T> &set);

	//@}

private:
	/**	\brief Computes the non-trivial path from node i to node j
			(j > i). Returns the length of the path or 0 if such
			path doesn't exist
	 **/
	size_t get_path(const branching &br, size_t i, size_t j,
		size_t (&path)[N]);

	/**	\brief Tests the membership of a %permutation in G_{i-1}
	 **/
	bool is_member(const branching &br, size_t i,
		const permutation<N> &perm);

	/**	\brief Computes a branching using a generating set; returns
			the generating set of G_{i-1}
	 **/
	void make_branching(branching &br, size_t i, const perm_list_t &gs,
		perm_list_t &gs2);

	void sift(const permutation<N> &perm);
};


template<size_t N, typename T>
const char *permutation_group<N, T>::k_clazz = "permutation_group<N, T>";


template<size_t N, typename T>
permutation_group<N, T>::permutation_group(
	const symmetry_element_set_adapter<N, T, se_perm_t> &set) {

	perm_list_t symm_gs1, symm_gs2, asymm_gs1, asymm_gs2;

	typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;
	for(typename adapter_t::iterator i = set.begin(); i != set.end(); i++) {

		const se_perm_t &e = set.get_elem(i);
		if(e.is_symm()) symm_gs1.push_back(e.get_perm());
		else asymm_gs1.push_back(e.get_perm());
	}

	perm_list_t *p1 = &symm_gs1, *p2 = &symm_gs2;	
	for(size_t i = 0; i < N; i++) {
		make_branching(m_symm, i, *p1, *p2);
		perm_list_t *t = p1; p1 = p2; p2 = t;
		p2->clear();
	}

	p1 = &asymm_gs1; p2 = &asymm_gs2;
	for(size_t i = 0; i < N; i++) {
		make_branching(m_asymm, i, *p1, *p2);
		perm_list_t *t = p1; p1 = p2; p2 = t;
		p2->clear();
	}
}


template<size_t N, typename T>
inline bool permutation_group<N, T>::is_member(
	const permutation<N> &perm, bool sign) {

	return is_member(sign ? m_symm : m_asymm, 0, perm);
}


template<size_t N, typename T>
void permutation_group<N, T>::convert(symmetry_element_set<N, T> &set) {

	static const char *method = "convert(symmetry_element_set<N, T>&)";

	//~ throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
}


template<size_t N, typename T>
size_t permutation_group<N, T>::get_path(
	const branching &br, size_t i, size_t j, size_t (&path)[N]) {

	if(j <= i) return 0;

	size_t p[N];

	register size_t k = j;
	register size_t len = 0;
	while(k != 0 && k != i) {
		p[len++] = k;
		k = br.m_edges[k];
	}
	if(k != i) return 0;

	for(k = 0; k < len; k++) path[k] = p[len - k - 1];

	return len;
}


template<size_t N, typename T>
bool permutation_group<N, T>::is_member(
	const branching &br, size_t i, const permutation<N> &perm) {

	std::cout << "is_member(" << i << ", " << perm << ")" << std::endl;

	if(perm.is_identity()) return true;
	if(i >= N - 1) return false;

	//	Find the element pi1 of the right coset representative U_{i+1}
	//	for which pi*pi1^{-1} stabilizes i. (pi == perm).
	permutation<N> rho;
	bool found = false;
	for(size_t j = i + 1; j < N; j++) {
		size_t path[N];
		size_t pathlen = get_path(br, i, j, path);
		if(pathlen == 0) continue;

		permutation<N> pi(perm), pi1;
		for(size_t k = 0; k < pathlen; k++)
			pi1.permute(br.m_labels[path[k]]);
		pi1.invert();
		pi.permute(pi1);
		std::cout << "path " << i << "->" << j << " (len = " <<
			pathlen << "): " << pi << std::endl;

		size_t seq[N];
		for(size_t k = 0; k < N; k++) seq[k] = k;
		pi.apply(seq);
		if(seq[i] == i) {
			rho.permute(pi);
			found = true;
			break;
		}
	}
	if(found) return is_member(br, i + 1, rho);
	else return false;
}


template<size_t N, typename T>
void permutation_group<N, T>::make_branching(branching &br, size_t i,
	const perm_list_t &gs, perm_list_t &gs2) {

	if(gs.empty()) return;

	perm_vec_t transv(N);

	std::cout << "transversal(" << i << ")" << std::endl;
	std::cout << "genset: <";
	for(typename perm_list_t::const_iterator pi = gs.begin();
		pi != gs.end(); pi++) {
		std::cout << " " << *pi;
	}
	std::cout << " >" << std::endl;

	std::vector<size_t> delta;
	delta.push_back(i);

	std::list<size_t> s;
	s.push_back(i);

	transv[i].reset();

	while(!s.empty()) {

		size_t j = s.front();
		s.pop_front();

		for(typename perm_list_t::const_iterator pi = gs.begin();
			pi != gs.end(); pi++) {

			size_t seq[N];
			for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
			pi->apply(seq);

			size_t k = seq[j];
			typename std::vector<size_t>::iterator dd = delta.begin();
			while(dd != delta.end() && *dd != k) dd++;
			if(dd == delta.end()) {
				permutation_t p(*pi);
				p.permute(transv[j]);
				transv[k].reset();
				transv[k].permute(p);
				delta.push_back(k);
				s.push_back(k);
			}
		}
	}

	std::cout << "transv: {";
	for(size_t j = 0; j < N; j++) std::cout << " " << transv[j];
	std::cout << " }" << std::endl;

	for(typename std::vector<size_t>::iterator dd = delta.begin();
		dd != delta.end(); dd++) {

		size_t j = *dd;
		if(j == i) continue;

		// add a new edge (remove an existing edge if necessary)
		br.m_edges[j] = i;
		br.m_labels[j].reset();
		br.m_labels[j].permute(transv[j]);
	}

	std::cout << "graph: {" << std::endl;
	for(size_t j = 0; j < N; j++) {
		size_t k = br.m_edges[j];
		if(k == N) continue;
		std::cout << k << " ==" << br.m_labels[j] << "==> " << j << std::endl;
	}
	std::cout << "}" << std::endl;

	for(typename perm_list_t::const_iterator pi = gs.begin();
		pi != gs.end(); pi++) {

		size_t seq[N];
		for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
		pi->apply(seq);

		for(typename std::vector<size_t>::iterator dd = delta.begin();
			dd != delta.end(); dd++) {

			size_t j = *dd, k = seq[j];
			permutation<N> p(transv[k], true);
			p.permute(*pi).permute(transv[j]);
			if(!p.is_identity()) gs2.push_back(p);
		}
	}

	std::cout << "genset2: <";
	for(typename perm_list_t::const_iterator pi = gs2.begin();
		pi != gs2.end(); pi++) {

		std::cout << " " << *pi;
	}
	std::cout << " >" << std::endl;

	//~ for(typename perm_list_t::const_iterator pi = gs2.begin();
		//~ pi != gs2.end(); pi++) {

		//~ sift(*pi);
	//~ }
}


template<size_t N, typename T>
void permutation_group<N, T>::sift(const permutation<N> &perm) {

}


} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_H