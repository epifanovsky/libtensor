#ifndef LIBTENSOR_PERMUTATION_GROUP_H
#define LIBTENSOR_PERMUTATION_GROUP_H

#include <algorithm>
#include <list>
#include "../defs.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "bad_symmetry.h"
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
	typedef permutation<N> perm_t;
	typedef std::pair<perm_t, bool> signed_perm_t;

	//!	Stores one labeled branching
	struct branching {
		signed_perm_t m_sigma[N]; //!< Edge labels (permutation + sign)
		signed_perm_t m_tau[N]; //!< Vertex labels (permutation + sign)
		size_t m_edges[N]; //!< Edge sources
		branching() {
			for(register size_t i = 0; i < N; i++) {
				m_edges[i] = N; m_sigma[i].second = m_tau[i].second = true;
			}
		}
		void reset() {
			for(register size_t i = 0; i < N; i++) {
				m_edges[i] = N;
				m_sigma[i].first.reset();
				m_tau[i].first.reset();
				m_sigma[i].second = m_tau[i].second = true;
			}
		}
	};

private:
	typedef se_perm<N, T> se_perm_t;

	typedef std::list<signed_perm_t> perm_list_t;
	typedef std::vector<signed_perm_t> perm_vec_t;

private:
	branching m_br; //!< Branching

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

	/**	\brief Augments the group with an %orbit represented by a
			%permutation.
		\param sign Symmetric (true)/anti-symmetric (false).
		\param perm Permutation.

		Does nothing if the subgroup with the same sign already contains
		the %orbit. Throws bad_symmetry if the subgroup with the
		opposite sign contains the %orbit.
	 **/
	void add_orbit(bool sign, const permutation<N> &perm);

	/**	\brief Tests the membership of a %permutation in the group
		\param sign Symmetric (true)/anti-symmetric (false).
		\param perm Permutation.
	 **/
	bool is_member(bool sign, const permutation<N> &perm) const;

	/**	\brief Converts the %permutation group to a generating set
			using the standard format
	 **/
	void convert(symmetry_element_set<N, T> &set) const;

	/**	\brief Generates a subgroup, all permutations in which
			stabilize unmasked elements. The %mask must have M
			masked elements for the operation to succeed.
	 **/
	template<size_t M>
	void project_down(const mask<N> &msk, permutation_group<M, T> &g2);

	/** \brief Generates a subgroup of all permutations that stabilize
			the set of N - M masked elements the %mask must have.
	 	\param msk Masks the elements to be stabilized (N - M times true)
		\param g2 Resulting subgroup
	 **/
	template<size_t M>
	void stabilize(const mask<N> &msk, permutation_group<M, T> &g2);

	/** \brief Generates a subgroup of all permutations that setwise stabilize
			the given K sets of masked elements. Each %mask must have different
			elements masked than any other. The total number of masked elements
			must be N - M.
	 	\param msk K masks of elements to be stabilized
		\param g2 Resulting subgroup
	 **/
	template<size_t M, size_t K>
	void stabilize(const mask<N> (&msk)[K], permutation_group<M, T> &g2);

	void permute(const permutation<N> &perm);

	//@}

private:
	/**	\brief Computes the non-trivial path from node i to node j
			(j > i). Returns the length of the path or 0 if such
			path doesn't exist
	 **/
	size_t get_path(const branching &br, size_t i, size_t j,
		size_t (&path)[N]) const;

	/**	\brief Tests the membership of a %permutation in G_{i-1}
			(or G for i==0)
	 **/
	bool is_member(const branching &br, size_t i,
		bool sign, const permutation<N> &perm) const;

	/**	\brief Computes a branching using a generating set; returns
			the generating set of G_{i-1}
	 **/
	void make_branching(branching &br, size_t i, const perm_list_t &gs,
		perm_list_t &gs2);

	void make_genset(const branching &br, perm_list_t &gs) const;

	void permute_branching(branching &br, const permutation<N> &perm);

	/**	\brief Computes a generating set for the subgroup that stabilizes
			one or more sets given by msk
		\param br Branching representing the group
		\param msk Sequence specifying the sets to stabilize
		\param gs Generating set for the subgroup
	 **/
	void make_setstabilizer(const branching &br, const sequence<N, size_t> &msk,
			perm_list_t &gs);

};


template<size_t N, typename T>
const char *permutation_group<N, T>::k_clazz = "permutation_group<N, T>";


template<size_t N, typename T>
permutation_group<N, T>::permutation_group(
	const symmetry_element_set_adapter<N, T, se_perm_t> &set) {

	perm_list_t gs1, gs2;

	typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;
//	std::cout << "ctor" << std::endl;
	for(typename adapter_t::iterator i = set.begin(); i != set.end(); i++) {

		const se_perm_t &e = set.get_elem(i);
//		std::cout << (e.is_symm() ? "symm" : "asymm") << " " << e.get_perm << std::endl;
		gs1.push_back(signed_perm_t(e.get_perm(), e.is_symm()));
	}

	perm_list_t *p1 = &gs1, *p2 = &gs2;
	for(size_t i = 0; i < N; i++) {
		make_branching(m_br, i, *p1, *p2);
		std::swap(p1, p2);
		p2->clear();
	}
//	std::cout << "Symm branching:" << std::endl;
//	for (size_t i = 0; i < N; i++) {
//		std::cout << i << " - " << m_symm.m_edges[i] << ": ";
//		std::cout << m_symm.m_sigma[i] << ", " << m_symm.m_tau[i] << std::endl;
//	}

//	std::cout << "ctor end" << std::endl;
}


template<size_t N, typename T>
void permutation_group<N, T>::add_orbit(bool sign, const permutation<N> &perm) {

	static const char *method = "add_orbit(bool, const permutation<N>&)";

	if(is_member(!sign, perm)) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"perm");
	}
	if(is_member(sign, perm)) return;

	perm_list_t gs1, gs2;
	make_genset(m_br, gs1);
	gs1.push_back(signed_perm_t(perm, sign));
	m_br.reset();

	perm_list_t *p1 = &gs1, *p2 = &gs2;
	for(size_t i = 0; i < N; i++) {
		make_branching(m_br, i, *p1, *p2);
		std::swap(p1, p2);
		p2->clear();
	}
}


template<size_t N, typename T>
inline bool permutation_group<N, T>::is_member(
	bool sign, const permutation<N> &perm) const {

	if(!sign && perm.is_identity()) return false;
	return is_member(m_br, 0, sign, perm);
}


template<size_t N, typename T>
void permutation_group<N, T>::convert(symmetry_element_set<N, T> &set) const {

	static const char *method = "convert(symmetry_element_set<N, T>&)";

	perm_list_t gs;

	make_genset(m_br, gs);
	for(typename perm_list_t::iterator i = gs.begin(); i != gs.end(); i++) {
		set.insert(se_perm_t(i->first, i->second));
	}
	gs.clear();
}


template<size_t N, typename T> template<size_t M>
void permutation_group<N, T>::project_down(
	const mask<N> &msk, permutation_group<M, T> &g2) {

	static const char *method =
		"project_down<M>(const mask<N>&, permutation_group<M, T>&)";

	register size_t m = 0;
	for(register size_t i = 0; i < N; i++) if(msk[i]) m++;
	if(m != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"msk");
	}

	branching br;
	perm_list_t gs1, gs2;
	perm_list_t *p1 = &gs1, *p2 = &gs2;
	make_genset(m_br, gs1);

	for(size_t i = 0; i < N; i++) {
		if(msk[i]) continue;
//		std::cout << "genset before branching of " << i << ": <";
//		for (typename perm_list_t::const_iterator pi = p1->begin();
//				pi != p1->end(); pi++)
//			std::cout << " " << *pi;
//		std::cout << ">" << std::endl;
		br.reset();
		make_branching(br, i, *p1, *p2);
		std::swap(p1, p2);
		p2->clear();
	}
//	std::cout << "genset1: <";
//	for(typename perm_list_t::const_iterator pi = p1->begin();
//		pi != p1->end(); pi++) {
//		std::cout << " " << *pi;
//	}
//	std::cout << " >" << std::endl;
//	std::cout << "genset2: <";
	for(typename perm_list_t::const_iterator pi = p1->begin();
		pi != p1->end(); pi++) {

		sequence<N, size_t> seq1a(0), seq2a(0);
		sequence<M, size_t> seq1b(0), seq2b(0);
		for(size_t i = 0; i < N; i++) seq2a[i] = seq1a[i] = i;
		pi->first.apply(seq2a);
		size_t j = 0;
		for(size_t i = 0; i < N; i++) {
			if(!msk[i]) continue;
			seq1b[j] = seq1a[i];
			seq2b[j] = seq2a[i];
			j++;
		}
		permutation_builder<M> pb(seq2b, seq1b);
//		std::cout << " " << pb.get_perm();
		g2.add_orbit(pi->second, pb.get_perm());
	}
//	std::cout << " >" << std::endl;

}


template<size_t N, typename T> template<size_t M>
void permutation_group<N, T>::stabilize(
	const mask<N> &msk, permutation_group<M, T> &g2) {

	static const char *method =
		"stabilize<M>(const mask<N>&, permutation_group<M, T>&)";

	mask<N> msks[1];
	msks[0] = msk;

	stabilize(msks, g2);
}

template<size_t N, typename T> template<size_t M, size_t K>
void permutation_group<N, T>::stabilize(
	const mask<N> (&msk)[K], permutation_group<M, T> &g2) {

	static const char *method =
		"stabilize<M>(const mask<N>&, permutation_group<N - M, T>&)";

	sequence<N, size_t> tm(0);
	register size_t nm = 0;
	for(register size_t k = 0; k < K; k++) {
		const mask<N> &msk_k = msk[k];
		for (register size_t i = 0; i < N; i++) {
			if (! msk_k[i]) continue;
			if (tm[i] != 0)
				throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
					"Index masked twice.");

			tm[i] = k + 1;
			nm++;
		}
	}
	if(nm != N - M)
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");

	// generating set of G(P)
	perm_list_t gs;
	make_setstabilizer(m_br, tm, gs);

	for (typename perm_list_t::const_iterator pi = gs.begin();
			pi != gs.end(); pi++) {

		sequence<N, size_t> seq1a(0), seq2a(0);
		sequence<M, size_t> seq1b(0), seq2b(0);
		for (size_t i = 0; i < N; i++) seq1a[i] = seq2a[i] = i;
		pi->first.apply(seq2a);
		for (size_t i = 0, j = 0; i < N; i++) {
			if (tm[i] != 0) continue;
			seq1b[j] = seq1a[i];
			seq2b[j] = seq2a[i];
			j++;
		}
		permutation_builder<M> pb(seq2b, seq1b);
		// if the resulting permutation is the identity just skip.
		if (pb.get_perm().is_identity())
			if (pi->second) continue;
			else throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
						"Illegal result permutation group.");

		g2.add_orbit(pi->second, pb.get_perm());

	}
	gs.clear();
}

template<size_t N, typename T>
void permutation_group<N, T>::permute(const permutation<N> &perm) {

	if(perm.is_identity()) return;

	permute_branching(m_br, perm);
}


template<size_t N, typename T>
size_t permutation_group<N, T>::get_path(
	const branching &br, size_t i, size_t j, size_t (&path)[N]) const {

	if(j <= i) return 0;

	size_t p[N];

	register size_t k = j;
	register size_t len = 0;
	while(k != N && k != i) {
		p[len++] = k;
		k = br.m_edges[k];
	}
	if(k != i) return 0;

	for(k = 0; k < len; k++) path[k] = p[len - k - 1];

	return len;
}


template<size_t N, typename T>
bool permutation_group<N, T>::is_member(const branching &br,
		size_t i, bool sign, const permutation<N> &perm) const {

	//~ std::cout << "is_member(" << i << ", " << perm << ")" << std::endl;

	sequence<N, size_t> seq(0);

	if(perm.is_identity()) return sign;
	if(i >= N - 1) return false;

	//	Find the element pi1 of the right coset representative Ui
	//	for which rho = pi * pi1^{-1} stabilizes i. (pi == perm).

	//	Check the identity first
	//
	for(size_t k = 0; k < N; k++) seq[k] = k;
	perm.apply(seq);
	if(seq[i] == i) {
		return is_member(br, i + 1, sign, perm);
	}

	//	Go over non-identity members of Ui
	//
	for(size_t j = i + 1; j < N; j++) {

		size_t path[N];
		size_t pathlen = get_path(br, i, j, path);
		if(pathlen == 0) continue;

		perm_t sigmaij(br.m_tau[j].first), tauiinv(br.m_tau[i].first, true);
		sigmaij.permute(tauiinv);
		
		perm_t rho, pi1inv(sigmaij, true);
		bool rho_sign;
		//~ std::cout << "path " << i << "->" << j << " (len = " <<
			//~ pathlen << "): sigmaij=" << sigmaij << " pi1inv=" << pi1inv;
		rho.permute(pi1inv);
		rho.permute(perm);
		rho_sign = ((br.m_tau[i].second == br.m_tau[j].second) ? sign : ! sign);
		//~ std::cout << " rho=" << rho << std::endl;

		for(size_t k = 0; k < N; k++) seq[k] = k;
		rho.apply(seq);
		if(seq[i] == i) {
			return is_member(br, i + 1, rho_sign, rho);
		}
	}
	return false;
}


template<size_t N, typename T>
void permutation_group<N, T>::make_branching(branching &br, size_t i,
	const perm_list_t &gs, perm_list_t &gs2) {

	if(gs.empty()) return;

	perm_vec_t transv(N);

//	std::cout << "make_branching" << std::endl;
//	std::cout << "transversal(" << i << ")" << std::endl;
//	std::cout << "genset: <";
//	for(typename perm_list_t::const_iterator pi = gs.begin();
//		pi != gs.end(); pi++) {
//		std::cout << " " << *pi;
//	}
//	std::cout << " >" << std::endl;

	std::vector<size_t> delta;
	delta.push_back(i);

	std::list<size_t> s;
	s.push_back(i);

	transv[i].first.reset();
	transv[i].second = true;

	while(! s.empty()) {

		size_t j = s.front();
		s.pop_front();

		for(typename perm_list_t::const_iterator pi = gs.begin();
			pi != gs.end(); pi++) {

			sequence<N, size_t> seq(0);
			for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
			pi->first.apply(seq);

			size_t k = seq[j];
			typename std::vector<size_t>::iterator dd = delta.begin();
			while(dd != delta.end() && *dd != k) dd++;
			if(dd == delta.end()) {
				signed_perm_t p(*pi);
				p.first.permute(transv[j].first);
				p.second = (transv[j].second ? p.second : ! p.second);
				transv[k].first.reset();
				transv[k].first.permute(p.first);
				transv[k].second = p.second;
				delta.push_back(k);
				s.push_back(k);
			}
		}
	}

//	std::cout << "transv: {";
//	for(size_t j = 0; j < N; j++) std::cout << " " << transv[j];
//	std::cout << " }" << std::endl;

	for(typename std::vector<size_t>::iterator dd = delta.begin();
		dd != delta.end(); dd++) {

		size_t j = *dd;
		if(j == i) continue;

		// add a new edge (remove an existing edge if necessary)
		br.m_edges[j] = i;
		br.m_sigma[j].first.reset();
		br.m_sigma[j].first.permute(transv[j].first);
		br.m_sigma[j].second = transv[j].second;
		br.m_tau[j].first.reset();
		br.m_tau[j].first.permute(br.m_sigma[j].first);
		br.m_tau[j].first.permute(br.m_tau[i].first);
		br.m_tau[j].second = (br.m_sigma[j].second ?
				br.m_tau[i].second : ! br.m_tau[i].second);
	}

//	std::cout << "graph: {" << std::endl;
//	for(size_t j = 0; j < N; j++) {
//		size_t k = br.m_edges[j];
//		if(k == N) continue;
//		permutation<N> pinv(br.m_sigma[j], true);
//		std::cout << k << "->" << j << " " << br.m_sigma[j] << " " <<
//				br.m_tau[j] << " " << j << "->" << k << " " << pinv << std::endl;
//	}
//	std::cout << "}" << std::endl;

	for(typename perm_list_t::const_iterator pi = gs.begin();
		pi != gs.end(); pi++) {

		sequence<N, size_t> seq(0);
		for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
		pi->first.apply(seq);

		for(typename std::vector<size_t>::iterator dd = delta.begin();
			dd != delta.end(); dd++) {

			size_t j = *dd, k = seq[j];
			signed_perm_t p(perm_t(transv[k].first, true), transv[k].second);
			p.first.permute(pi->first).permute(transv[j].first);
			p.second =
					((pi->second == transv[j].second) ? p.second : ! p.second);
			if(! p.first.is_identity()) {
				typename perm_list_t::const_iterator rho =
					gs2.begin();
				while(rho != gs2.end() && ! rho->first.equals(p.first)) rho++;
				if(rho == gs2.end()) gs2.push_back(p);
			}
			else if (! p.second) {
				throw generic_exception(g_ns, k_clazz,
						"make_branching(branching, size_t, "
						"const perm_list_t &, perm_list_t&)",
						__FILE__, __LINE__,
						"Illegal permutation.");
			}
		}
	}

//	std::cout << "genset2: <";
//	for(typename perm_list_t::const_iterator pi = gs2.begin();
//			pi != gs2.end(); pi++) {
//
//		std::cout << " " << *pi;
//	}
//	std::cout << " >" << std::endl;

}


template<size_t N, typename T>
void permutation_group<N, T>::make_genset(
	const branching &br, perm_list_t &gs) const {

	for(register size_t i = 0; i < N; i++) {
		if(br.m_edges[i] != N && ! br.m_sigma[i].first.is_identity()) {
			gs.push_back(br.m_sigma[i]);
		}
	}
	//~ std::cout << "genset: <";
	//~ for(typename perm_list_t::const_iterator pi = gs.begin();
		//~ pi != gs.end(); pi++) {

		//~ std::cout << " " << *pi;
	//~ }
	//~ std::cout << " >" << std::endl;
}


template<size_t N, typename T>
void permutation_group<N, T>::permute_branching(
	branching &br, const permutation<N> &perm) {

	//~ std::cout << "graph(bef): {" << std::endl;
	//~ for(size_t j = 0; j < N; j++) {
		//~ size_t k = br.m_edges[j];
		//~ if(k == N) continue;
		//~ permutation<N> pinv(br.m_sigma[j], true);
		//~ std::cout << k << "->" << j << " " << br.m_sigma[j] << " " << br.m_tau[j]
			//~ << " " << j << "->" << k << " " << pinv << std::endl;
	//~ }
	//~ std::cout << "}" << std::endl;
	//~ std::cout << "perm: " << perm << std::endl;

	perm_list_t gs1, gs2, gs3;
	make_genset(br, gs1);
	for(typename perm_list_t::iterator i = gs1.begin();
		i != gs1.end(); i++) {

		sequence<N, size_t> seq1(0), seq2(0);
		for(size_t j = 0; j < N; j++) seq2[j] = seq1[j] = j;
		i->first.apply(seq2);
		permutation_builder<N> pb(seq2, seq1, perm);
		gs2.push_back(signed_perm_t(pb.get_perm(), i->second));
	}
	br.reset();
	perm_list_t *p1 = &gs2, *p2 = &gs3;
	for(size_t i = 0; i < N; i++) {
		make_branching(br, i, *p1, *p2);
		std::swap(p1, p2);
		p2->clear();
	}

	//~ std::cout << "graph(aft): {" << std::endl;
	//~ for(size_t j = 0; j < N; j++) {
		//~ size_t k = br.m_edges[j];
		//~ if(k == N) continue;
		//~ permutation<N> pinv(br.m_sigma[j], true);
		//~ std::cout << k << "->" << j << " " << br.m_sigma[j] << " " << br.m_tau[j]
			//~ << " " << j << "->" << k << " " << pinv << std::endl;
	//~ }
	//~ std::cout << "}" << std::endl;
}

template<size_t N, typename T>
void permutation_group<N, T>::make_setstabilizer(
	const branching &br, const sequence<N, size_t> &msk, perm_list_t &gs) {

	static const char *method =
		"make_set_stabilizer(const branching &, const mask<N>&, perm_list_t&)";

	register size_t m = 0;
	for(register size_t i = 0; i < N; i++) if(msk[i] != 0) m++;
	if(m == 0) {
		make_genset(br, gs);
		return;
	}
	if(m == 1) {
		branching brx;
		perm_list_t gsx;
		make_genset(br, gsx);

		size_t i = 0;
		for(; i < N; i++) if(msk[i]) break;

		make_branching(brx, i, gsx, gs);
		return;
	}

	// loop over all stabilizers G_i starting from G_{N-1} since G_N = <()>
	for (size_t i = N - 1, ii = i - 1; i > 0; i--, ii--) {
		// ii is the proper index!!!

		// we need N - ii permutations to build permutation
		// g = u_{N - 1} ... u_{ii}
		perm_vec_t pu(N - i); // vector u with u_i \in U_i
		std::vector<size_t> ui(N - i);
		// initialize ui and pu
		for (size_t k = ii; k < N - 1; k++) ui[k - ii] = k;
		ui[0]++;

		for (; ui[0] < N; ui[0]++) {

			size_t k = br.m_edges[ui[0]];
			while (k != ii && k != N)
				k = br.m_edges[k];

			if (k == N) continue;

			pu[0].first.reset();
			pu[0].first.permute(br.m_tau[ui[0]].first);
			pu[0].first.permute(perm_t(br.m_tau[k].first, true));
			pu[0].second = (br.m_tau[ui[0]].second == br.m_tau[k].second);
			break;
		}

		// loop over all possible sequences u_{N-1} ... u_{ii}
		while (ui[0] != N) {

			signed_perm_t g;
			g.second = true;

			// build the permutation g = u_{N-1} ... u_{ii}
			for (size_t k = 0; k < ui.size(); k++) {
				if (ui[k] == k + ii) continue;

				g.first.permute(pu[k].first);
				g.second = (pu[k].second ? g.second : ! g.second);
			}

			// check whether g is in G(P)
			sequence<N, size_t> seq(0);
			for (size_t k = 0; k < N; k++) seq[k] = k;
			g.first.apply(seq);
			size_t l = 0;
			for (; l < N; l++)
				if (msk[l] != msk[seq[l]]) break;

			// if g is in G(P), we add it to the list of permutations,
			// skip this level ii and go to the next level
			if (l == N)	{
				gs.push_back(g);
				break;
			}

			// else go to the next sequence u_{N-1} ... u_ii
			for (size_t k = ui.size(), k1 = k - 1; k > 0; k--, k1--) {
				ui[k1]++;
				for (; ui[k1] < N; ui[k1]++) {

					size_t m = br.m_edges[ui[k1]];
					while (m != ii + k1 && m != N) m = br.m_edges[m];

					if (m == N) continue;

					pu[k1].first.reset();
					pu[k1].first.permute(br.m_tau[ui[k1]].first);
					pu[k1].first.permute(perm_t(br.m_tau[m].first, true));
					pu[k1].second =
							(br.m_tau[ui[k1]].second == br.m_tau[m].second);
					break;
				}

				if (ui[k1] != N || k1 == 0) break;

				ui[k1] = ii + k1;
				pu[k1].first.reset();
				pu[k1].second = true;
			}
		}

		// if we broke off in the middle since we found a proper g
		// we still have to test all elements G_i \ G_{i+1}
		while (ui[0] != N) {
			ui[0]++;
			for (; ui[0] < N; ui[0]++) {

				size_t k = br.m_edges[ui[0]];
				while (k != ii && k != N) k = br.m_edges[k];

				if (k == N) continue;

				pu[0].first.reset();
				pu[0].first.permute(br.m_tau[ui[0]].first);
				pu[0].first.permute(perm_t(br.m_tau[k].first, true));
				pu[0].second =
						(br.m_tau[ui[0]].second == br.m_tau[k].second);
				break;
			}
			if (ui[0] == N) break;

			// here g = u_i
			signed_perm_t &g = pu[0];
			// check whether g is in G(P)
			sequence<N, size_t> seq(0);
			for (size_t k = 0; k < N; k++) seq[k] = k;
			g.first.apply(seq);
			size_t l = 0;
			for (; l < N; l++)
				if (msk[l] && ! msk[seq[l]]) break;

			if (l == N)	gs.push_back(g);
		}
	}
}



} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_H
