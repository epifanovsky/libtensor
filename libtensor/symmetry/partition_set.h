#ifndef LIBTENSOR_PARTITION_SET_H
#define LIBTENSOR_PARTITION_SET_H

#include <algorithm>
#include <list>
#include <map>
#include "../defs.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "bad_symmetry.h"
#include "symmetry_element_set_adapter.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Representation of a set of partitions

	Implements the representation of a set of partitions as a map of se_part
	objects, each uniquely identified by the number of partitions along the
	dimensions.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class partition_set {
public:
	static const char *k_clazz; //!< Class name

private:
	typedef se_part<N, T> se_part_t;
	typedef symmetry_element_set_adapter<N, T, se_part_t> adapter_t;

	typedef std::map<size_t, se_part_t *> map_t;
	typedef std::pair<size_t, se_part_t *> pair_t;

private:
	block_index_space<N> m_bis; //!< Block index space
	map_t m_map; //!< Map npart -> se_part

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates an empty partition set
	 	\param bis Block index space.
	 **/
	partition_set(const block_index_space<N> &bis) :
		m_bis(bis) { }

	/**	\brief Creates a partition set from set of se_part
		\param set Set of se_part objects.
	 **/
	partition_set(const adapter_t &set);

	/**	\brief Creates a partition set from set of se_part
		\param set Set of se_part objects.
		\param perm Permutation.
	 **/
	partition_set(const adapter_t &set, const permutation<N> &perm);

	/**	\brief Destroys the object
	 **/
	~partition_set() { clear(); }

	//@}

	//! \name Access functions
	//@{

	/**	\brief Return the block index space of the partition set.
	 **/
	const block_index_space<N> &get_bis() const { return m_bis; }
	//@}

	//!	\name Manipulations
	//@{

	/**	\brief Augments the set with another partition.
		\param part Partition.
		\param perm Permutation.
	 **/
	void add_partition(const se_part<N, T> &part,
			const permutation<N> &perm);

	/**	\brief Augments the set with a partition having less dimensions.
		\param part Partition.
		\param perm Permutation.
	 	\param msk Mask.
	 **/
	template<size_t M>
	void add_partition(const se_part<M, T> &part,
			const permutation<M> &perm, const mask<N> &msk);

	/** \brief Clears the set.
	 **/
	void clear();

	/** \brief Permutes the partition set
	 **/
	void permute(const permutation<N> &perm);

	/** \brief Computes the intersection with another partition_set.
	 	\param set Partition set.
	 	\param mult Indicates multiplication (see below).

	 	The current partition set is modified so that only those partitions
	 	remain that are also present in the other set. Within the remaining
	 	partitions only those mappings are kept that are present in the
	 	respective partition of the other set. However, the exact handling of
	 	mappings that are present in both sets depends on the mult flag passed
	 	to this function. If mult = false, two mappings are considered
	 	identical only if both indexes and signs are identical. The result
	 	mapping then has the sign of both mappings. If mult = true, only the
	 	indexes of the mappings have to be identical. The sign of the result
	 	mapping is then determined by both signs of the mappings, i.e. if both
	 	signs are identical, the resulting sign is positive (true), or if the
	 	signs are opposite, the resuling sign is negative (false).
	 **/
	void intersect(const partition_set<N, T> &set, bool mult = false);

	/** \brief Merges N - M + 1 dimensions given by mask into one.
		\param msk Mask.
	 	\param set Result partition set.
	 **/
	template<size_t M>
	void merge(const mask<N> &msk, partition_set<M, T> &set) const;

	/** \brief Stabilizes N - M dimensions given by the masks and removes them.
		\param msk Masks.
	 	\param set Result partition set.
	 **/
	template<size_t M, size_t K>
	void stabilize(const mask<N> (&msk)[K], partition_set<M, T> &set) const;

	/**	\brief Converts the partition set into a symmetry element set.
		\param set Resulting set.
	 **/
	void convert(symmetry_element_set<N, T> &set) const;


	//@}
private:
	/** \brief Transfers mappings from one se_part object to another
		\param from Source of mappings
	 	\param perm Permutation of the source mappings.
 	 	\param to Destination of mappings
	 **/
	static void transfer_mappings(const se_part<N, T> &from,
			const permutation<N> &perm, se_part<N, T> &to);

	/** \brief Transfers mappings from a se_part object with less dimensions to
	   	   one with more.
		\param from Source of mappings
	 	\param perm Permutation of the source mappings.
	 	\param msk Mask of affected dimension
 	 	\param to Destination of mappings
	 **/
	template<size_t M>
	static void transfer_mappings(
			const se_part<M, T> &from, const permutation<M> &perm,
			const mask<N> &msk, se_part<N, T> &to);

	static block_index_space<N> make_bis(const adapter_t&set);
};


template<size_t N, typename T>
const char *partition_set<N, T>::k_clazz = "partition_set<N, T>";

template<size_t N, typename T>
partition_set<N, T>::partition_set(const adapter_t &set) :
	m_bis(make_bis(set)) {

	permutation<N> perm;
	for (typename adapter_t::iterator it = set.begin();
			it != set.end(); it++) {
		const se_part_t &p = set.get_elem(it);
		add_partition(p, perm);
	}
}

template<size_t N, typename T>
partition_set<N, T>::partition_set(const adapter_t &set,
		const permutation<N> &perm) : m_bis(make_bis(set)) {

	m_bis.permute(perm);
	for (typename adapter_t::iterator it = set.begin();
			it != set.end(); it++) {
		const se_part_t &p = set.get_elem(it);
		add_partition(p, perm);
	}
}

template<size_t N, typename T>
void partition_set<N, T>::add_partition(
		const se_part_t &part, const permutation<N> &perm) {

	static const char *method =
			"add_partition(const se_part<N,T> &, const permutation<N>&)";

	// check block index space
	block_index_space<N> bis(part.get_bis());
	bis.permute(perm);
	if (! m_bis.equals(bis)) {
		throw bad_symmetry(g_ns, k_clazz,
				method, __FILE__, __LINE__, "Illegal block index space.");
	}

	// determine nparts
	size_t npart = part.get_npart();

	// find partitioning with nparts in map
	typename map_t::iterator it = m_map.find(npart);
	if (it == m_map.end()) {
		// Check if this partitioning is valid
		typename map_t::iterator it2 = m_map.begin();
		if (it2 != m_map.end()) {
			if (npart % it2->first != 0 && it2->first % npart != 0)
				throw bad_symmetry(g_ns, k_clazz, method,
						__FILE__, __LINE__, "Illegal partitioning.");
		}
		m_map[npart] = new se_part<N, T>(part);
		m_map[npart]->permute(perm);
	}
	else {
		mask<N> ma(part.get_mask());
		ma.permute(perm);

		se_part<N, T> &cur_part = *(it->second);
		const mask<N> &mb = cur_part.get_mask();

		if (ma.equals(mb)) {
			transfer_mappings(part, perm, cur_part);
		}
		else {
			mask<N> m(mb);
			m|=ma;

			if (m.equals(mb)) {
				transfer_mappings(part, perm, cur_part);
			}
			else {
				se_part<N, T> *new_part = new se_part<N, T>(m_bis, m, npart);
				transfer_mappings(cur_part, permutation<N>(), *new_part);
				transfer_mappings(part, perm, *new_part);

				delete it->second;
				it->second = new_part;
			}
		}
	}
}

template<size_t N, typename T> template<size_t M>
void partition_set<N, T>::add_partition(const se_part<M, T> &part,
		const permutation<M> &perm, const mask<N> &msk) {

	static const char *method = "add_partition("
			"const se_part<M,T> &, const permutation<M>&, const mask<N> &)";

	// check mask
	size_t m = 0;
	for (size_t i = 0; i < N; i++) if (msk[i]) m++;
	if (m != M) {
		throw bad_parameter(g_ns, k_clazz,
				method, __FILE__, __LINE__, "Illegal mask.");
	}

	// check block index space (but how?)

	// determine nparts
	size_t npart = part.get_npart();

	mask<M> ma(part.get_mask());
	ma.permute(perm);

	// find partitioning with nparts in map
	typename map_t::iterator it = m_map.find(npart);
	if (it == m_map.end()) {
		// Check if this partitioning is valid
		typename map_t::iterator it2 = m_map.begin();
		if (it2 != m_map.end()) {
			if (npart % it2->first != 0 && it2->first % npart != 0)
				throw bad_symmetry(g_ns, k_clazz, method,
						__FILE__, __LINE__, "Illegal partitioning.");
		}

		mask<N> mb(msk);
		for (size_t i = 0, j = 0; i < N; i++) {
			if (msk[i]) mb[i] = ma[j++];
		}

		se_part<N, T> *new_part = new se_part<N, T>(m_bis, mb, npart);
		transfer_mappings<M>(part, perm, msk, *new_part);
		m_map[npart] = new_part;
	}
	else {
		se_part<N, T> &cur_part = *(it->second);

		const mask<N> &mb = cur_part.get_mask();
		mask<N> mc(mb);
		for (size_t i = 0, j = 0; i < N; i++) {
			if (msk[i]) mc[i] = ma[j++] || mb[i];
		}

		if (mc.equals(mb)) {
			transfer_mappings(part, perm, msk, cur_part);
		}
		else {
			se_part<N, T> *new_part = new se_part<N, T>(m_bis, mc, npart);
			transfer_mappings(cur_part, permutation<N>(), *new_part);
			transfer_mappings(part, perm, msk, *new_part);

			delete it->second;
			it->second = new_part;
		}
	}
}

template<size_t N, typename T>
void partition_set<N, T>::clear() {

	for (typename map_t::iterator it = m_map.begin(); it != m_map.end(); it++) {
		delete it->second;
		it->second = 0;
	}
	m_map.clear();
}

template<size_t N, typename T>
void partition_set<N, T>::permute(const permutation<N> &perm) {

	for (typename map_t::iterator it = m_map.begin(); it != m_map.end(); it++) {

		it->second->permute(perm);
	}
}

template<size_t N, typename T>
void partition_set<N, T>::intersect(const partition_set<N, T> &set, bool mult) {

	if (! m_bis.equals(set.m_bis)) {
		throw bad_symmetry(g_ns, k_clazz,
				"intersect(const partition_set<N, T> &)",
				__FILE__, __LINE__, "set.m_bis");
	}

	typename map_t::iterator it1 = m_map.begin();
	size_t npart = it1->first;

	while (it1 != m_map.end()) {

		typename map_t::const_iterator it2 = set.m_map.find(npart);

		if (it2 == set.m_map.end()) {
			delete it1->second;
			m_map.erase(it1);
		}
		else {
			se_part<N, T> *x1, *x2;
			mask<N> m(it1->second->get_mask());
			if (m.equals(it2->second->get_mask())) {
				x1 = it1->second;
				x2 = it2->second;
			}
			else {
				m |= it2->second->get_mask();

				x1 = new se_part<N, T>(m_bis, m, npart);
				x2 = new se_part<N, T>(m_bis, m, npart);

				permutation<N> perm;
				transfer_mappings(*it1->second, perm, *x1);
				transfer_mappings(*it2->second, perm, *x2);
			}

			se_part<N, T> *new_part =
					new se_part<N, T>(m_bis, m, npart);

			bool empty = true;
			abs_index<N> ai(x1->get_pdims());
			do {
				const index<N> &i1 = ai.get_index();
				index<N> i2 = x1->get_direct_map(i1);
				while (! i2.equals(i1)) {
					if (x2->map_exists(i1, i2)) {
						bool sign1 = x1->get_sign(i1, i2);
						bool sign2 = x2->get_sign(i1, i2);
						permutation<N> p1(x1->get_perm(i1, i2));
						permutation<N> p2(x2->get_perm(i1, i2));

						if (mult) {
							if (p1.equals(p2)) {
								new_part->add_map(i1, i2, p2, sign1 == sign2);
								empty = false;
							}
						}
						else {
							if (sign1 == sign2 && p1.equals(p2)) {
								new_part->add_map(i1, i2, p1, sign1);
								empty = false;
							}
						}
						break;
					}
					i2 = x1->get_direct_map(i2);
				}

			} while (ai.inc());

			if (! m.equals(it2->second->get_mask())) {
				delete x1; delete x2;
			}

			delete it1->second;
			it1->second = new_part;

			if (empty) {
				delete new_part;
				m_map.erase(it1);
			}
		}

		npart++;
		it1 = m_map.find(npart);
	}
}

template<size_t N, typename T> template<size_t M, size_t K>
void partition_set<N, T>::stabilize(
		const mask<N> (&msk)[K], partition_set<M, T> &set) const {

	static const char *method =
			"stabilize<K>(const mask<N> &[K], partition_set<M, T> &)";

	typedef std::list< index<N> > list_t;

	// Check masks
	mask<N> tm;
	for (size_t k = 0; k < K; k++) tm |= msk[k];

	size_t m = 0;
	for (size_t i = 0; i < N; i++) if (tm[i]) m++;
	if (m != N - M)
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk.");

	sequence<N, size_t> map(0); // Map
	for (size_t i = 0, j = 0; i < N; i++) {
		if (tm[i]) {
			size_t k = 0;
			for (; k < K; k++) if (msk[k][i]) break;
			map[i] = M + k;
			continue;
		}
		map[i] = j++;
	}

	set.clear();
	permutation<M> perm;

	// loop over all se_parts
	for (typename map_t::const_iterator it = m_map.begin();
			it != m_map.end(); it++) {

		size_t npart = it->first;
		se_part<N, T> &pa1 = *(it->second);
		const mask<N> &ma1 = pa1.get_mask();

		// check if a dimensions which is part of the mapping is not stabilized
		size_t i = 0;
		for (; i < N; i++) if (ma1[i] && ! tm[i]) break;
		if (i == N) continue;

		// if necessary, extend the partition such that dimensions stabilized
		// together have the same partitioning
		index<K> i1, i2;
		mask<N> ma2(ma1);
		for (size_t k = 0; k < K; k++) {
			size_t j = 0;
			for (; j < N; j++) if (msk[k][j] && ma1[j]) break;
			if (j != N) {
				i2[k] = npart - 1;
				for (j = 0; j < N; j++) if (msk[k][j]) ma2[j] = true;
			}
		}

		se_part<N, T> *pa = 0;
		if (ma2.equals(ma1)) {
			pa = &pa1;
		}
		else {
			pa = new se_part<N, T>(pa1.get_bis(), ma2, npart);
			transfer_mappings(pa1, permutation<N>(), *pa);
		}

		dimensions<K> mdims(index_range<K>(i1, i2));

		// determine mask for result
		mask<M> mm;
		for (size_t i = 0; i < N; i++) if (! tm[i]) mm[map[i]] = ma1[i];

		// loop over all possible result mappings
		se_part<M, T> pb(set.get_bis(), mm, npart);
		bool empty = true;

		// loop over first map index
		abs_index<M> ai1(pb.get_pdims());
		do {
			const index<M> &idxb1 = ai1.get_index();

			// create a list of possible indexes from the input
			list_t l1;

			index<N> idxa1;
			for (size_t i = 0; i < N; i++) {
				if (tm[i]) continue;
				idxa1[i] = idxb1[map[i]];
			}
			l1.push_back(idxa1);

			abs_index<K> aim1(mdims);
			while (aim1.inc()) {
				const index<K> &idxk = aim1.get_index();
				for (size_t i = 0; i < N; i++) {
					if (! tm[i]) continue;
					idxa1[i] = idxk[map[i] - M];
				}
				l1.push_back(idxa1);
			}

			// loop over second map index
			abs_index<M> ai2(idxb1, pb.get_pdims());
			while (ai2.inc()) {
				const index<M> &idxb2 = ai2.get_index();

				// create a list of possible indexes from the input
				list_t l2;

				index<N> idxa2;
				for (size_t i = 0; i < N; i++) {
					if (tm[i]) continue;
					idxa2[i] = idxb2[map[i]];
				}

				l2.push_back(idxa2);
				abs_index<K> aim2(mdims);
				while (aim2.inc()) {
					const index<K> &idxk = aim2.get_index();
					for (size_t i = 0; i < N; i++) {
						if (! tm[i]) continue;
						idxa2[i] = idxk[map[i] - M];
					}
					l2.push_back(idxa2);
				}

				bool found = false, sign = true;
				// check if there is a 1-to-1 mapping of list l1 and l2
				typename list_t::iterator it1 = l1.begin();
				for (; it1 != l1.end(); it1++) {

					typename list_t::iterator it2 = l2.begin();
					for ( ; it2 != l2.end(); it2++) {
						if (pa->map_exists(*it1, *it2)) break;
					}

					if (it2 == l2.end()) break;

					bool s = pa->get_sign(*it1, *it2);
					if (found && sign != s) break;

					sign = s;
					found = true;
					l2.erase(it2);
				}
				if (it1 != l1.end()) continue;

				pb.add_map(idxb1, idxb2, sign);
				empty = false;
			}
		} while (ai1.inc());

		if (! empty) set.add_partition(pb, perm);

		if (! ma1.equals(ma2)) delete pa;
		pa = 0;
	}
}

template<size_t N, typename T> template<size_t M>
void partition_set<N, T>::merge(
		const mask<N> &msk, partition_set<M, T> &set) const {

	static const char *method =
			"stabilize<K>(const mask<N> &[K], partition_set<M, T> &)";

	// Check masks
	size_t m = 0;
	for (size_t i = 0; i < N; i++) if (msk[i]) m++;
	if (m != N - M + 1)
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk.");

	sequence<N, size_t> map(0); // Map
	m = 0;
	for (size_t i = 0, j = 0; i < N; i++, m++) {
		map[i] = i;
		if (msk[i]) break;
	}
	for (size_t i = m + 1, j = m + 1; i < N; i++) {
		if (msk[i]) map[i] = m;
		else map[i] = j++;
	}

	set.clear();
	permutation<M> perm;

	// loop over all se_parts
	for (typename map_t::const_iterator it1 = m_map.begin();
			it1 != m_map.end(); it1++) {

		size_t npart = it1->first;
		se_part<N, T> &pa = *(it1->second);
		const mask<N> &ma = pa.get_mask();

		// check if all masked dimensions are part of the mapping
		size_t i = 0;
		for (; i < N; i++) if (msk[i] && ! ma[i]) break;
		if (i != N) continue;

		// determine mask for result
		mask<M> mm;
		bool done = false;
		for (size_t i = 0, j = 0; i < N; i++) {
			if (msk[i] && done) continue;
			if (msk[i]) done = true;
			mm[j++] = ma[i];
		}

		// loop over all possible result mappings
		se_part<M, T> pb(set.get_bis(), mm, npart);
		bool empty = true;
		abs_index<M> ai1(pb.get_pdims());
		do {
			const index<M> &idxb1 = ai1.get_index();

			// determine input index
			index<N> idxa1;
			for (size_t i = 0; i < N; i++) idxa1[i] = idxb1[map[i]];

			abs_index<M> ai2(idxb1, pb.get_pdims());
			ai2.inc();
			do {
				const index<M> &idxb2 = ai2.get_index();

				// determine input index
				index<N> idxa2;
				for (size_t i = 0; i < N; i++) idxa2[i] = idxb2[map[i]];

				if (! pa.map_exists(idxa1, idxa2)) continue;

				bool sign = pa.get_sign(idxa1, idxa2);
				pb.add_map(idxb1, idxb2, sign);
				empty = false;

			} while (ai2.inc());
		} while (ai1.inc());

		if (! empty) set.add_partition(pb, perm);
	}
}

template<size_t N, typename T>
void partition_set<N, T>::convert(symmetry_element_set<N, T> &set) const {

	for (typename map_t::const_iterator it = m_map.begin();
			it != m_map.end(); it++) {

		set.insert(*it->second);
	}
}

template<size_t N, typename T>
void partition_set<N, T>::transfer_mappings(const se_part<N, T> &from,
		const permutation<N> &perm, se_part<N, T> &to) {

	static const char *method = "transfer_mappings(const se_part<N, T>&, "
			"const permutation<N> &, se_part<N, T>&)";

#ifdef LIBTENSOR_DEBUG
	if (from.get_npart() != to.get_npart()) {
		throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Incompatible se_part.");
	}
#endif

	mask<N> ma(from.get_mask()); ma.permute(perm);
	const mask<N> &mb = to.get_mask();

	// simple case: both dimensions are identical
	if (ma.equals(mb)) {

		abs_index<N> ai(from.get_pdims());
		do {
			index<N> idx1(ai.get_index());
			index<N> idx2(from.get_direct_map(idx1));

			if (idx1.equals(idx2)) continue;

			bool sx = from.get_sign(idx1, idx2);
			permutation<N> px(from.get_perm(idx1, idx2));

			idx1.permute(perm);
			idx2.permute(perm);

			if (! perm.is_identity()) {
				sequence<N, size_t> seq1a(0), seq2a(0);
				for (size_t i = 0; i < N; i++) seq1a[i] = seq2a[i] = i;
				seq1a.permute(perm);
				seq2a.permute(px); seq2a.permute(perm);
				permutation_builder<N> pb(seq2a, seq1a);
				px.reset();
				px.permute(pb.get_perm());
			}

			to.add_map(idx1, idx2, px, sx);

		} while (ai.inc());

	}
	else {
		mask<N> m(mb);
		m |= ma;

#ifdef LIBTENSOR_DEBUG
		if (! m.equals(mb)) {
			throw bad_symmetry(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Incompatible se_part.");
		}
#endif

		size_t npart = to.get_npart();
		abs_index<N> ai(from.get_pdims());
		do { // while (ai.inc())
			index<N> idx1(ai.get_index());
			index<N> idx2(from.get_direct_map(idx1));

			if (abs_index<N>(idx2, from.get_pdims()).get_abs_index()
					<= ai.get_abs_index()) continue;

			bool sx = from.get_sign(idx1, idx2);
			permutation<N> px(from.get_perm(idx1, idx2));

			idx1.permute(perm);
			idx2.permute(perm);

			if (! perm.is_identity()) {
				sequence<N, size_t> seq1a(0), seq2a(0);
				for (size_t i = 0; i < N; i++) seq1a[i] = seq2a[i] = i;
				seq1a.permute(perm);
				seq2a.permute(px); seq2a.permute(perm);
				permutation_builder<N> pb(seq2a, seq1a);
				px.reset();
				px.permute(pb.get_perm());
			}

			bool done = false;
			while (! done) {
				to.add_map(idx1, idx2, px, sx);

				// determine next index
				done = true;
				for (size_t i = N, j = N - 1; i > 0; i--, j--) {
					if (! mb[j] || ma[j]) continue;

					idx1[j]++;
					idx2[j]++;

					if (idx1[j] == npart) idx1[j] = idx2[j] = 0;
					else { done = false; break; }
				} // end for
			} // end while (!done)
		} while (ai.inc());
	}
}

template<size_t N, typename T> template<size_t M>
void partition_set<N, T>::transfer_mappings(const se_part<M, T> &from,
		const permutation<M> &perm, const mask<N> &msk, se_part<N, T> &to) {


	static const char *method = "transfer_mappings(const se_part<N, T>&, "
			"const permutation<N> &, const mask<N> &msk, se_part<N, T>&)";

#ifdef LIBTENSOR_DEBUG
	if (from.get_npart() != to.get_npart()) {
		throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Incompatible se_part.");
	}
#endif

	mask<M> mx(from.get_mask()); mx.permute(perm);
	mask<N> ma(msk);
	for (size_t i = 0, j = 0; i < N; i++) if (msk[i]) ma[i] = mx[j++];
	const mask<N> &mb = to.get_mask();

	size_t npart = to.get_npart();
	abs_index<M> ai(from.get_pdims());
	do {
		index<M> idx1a(ai.get_index()), idx2a(from.get_direct_map(idx1a));

		if (abs_index<M>(idx2a, from.get_pdims()).get_abs_index()
				<= ai.get_abs_index()) continue;

		bool sx = from.get_sign(idx1a, idx2a);
		permutation<M> px(from.get_perm(idx1a, idx2a));

		idx1a.permute(perm);
		idx2a.permute(perm);

		sequence<M, size_t> seq1a(0), seq2a(0);
		for (size_t i = 0; i < M; i++) seq1a[i] = seq2a[i] = i;
		seq1a.permute(perm);
		seq2a.permute(px); seq2a.permute(perm);

		index<N> idx1b, idx2b;
		sequence<N, size_t> seq1b(0), seq2b(0);
		for (size_t i = 0, j = 0; i < N; i++) {
			seq1b[i] = seq2b[i] = i;
			if (msk[i]) {
				idx1b[i] = idx1a[j]; idx2b[i] = idx2a[j];
				seq1b[i] = N + seq1a[j]; seq2b[i] = N + seq2a[j];
				j++;
			}
		}
		permutation_builder<N> pb(seq2b, seq1b);

		bool done = false;
		while (! done) {
			to.add_map(idx1b, idx2b, pb.get_perm(), sx);

			// determine next index
			done = true;
			for (size_t i = N, j = N - 1; i > 0; i--, j--) {
				if (! mb[j] || ma[j]) continue;

				idx1b[j]++;
				idx2b[j]++;

				if (idx1b[j] == npart) idx1b[j] = idx2b[j] = 0;
				else { done = false; break; }
			}
		}
	} while (ai.inc());
}

template<size_t N, typename T>
block_index_space<N> partition_set<N, T>::make_bis(const adapter_t &set) {

	typename adapter_t::iterator it = set.begin();
	if (it == set.end())
		throw bad_parameter(g_ns, k_clazz, "make_bis(const adapter_t &)",
				__FILE__, __LINE__, "Empty set.");

	return set.get_elem(it).get_bis();
}



} // namespace libtensor

#endif // LIBTENSOR_PARTITION_SET_H
