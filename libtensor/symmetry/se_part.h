#ifndef LIBTENSOR_SE_PART_H
#define LIBTENSOR_SE_PART_H

#include <algorithm>
#include "../defs.h"
#include "../core/abs_index.h"
#include "../core/block_index_space.h"
#include "../core/mask.h"
#include "../core/permutation_builder.h"
#include "../core/symmetry_element_i.h"
#include "../core/transf.h"
#include "bad_symmetry.h"

namespace libtensor {


/**	\brief Symmetry between %tensor partitions
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This %symmetry element establishes relationships between partitions
	of a block %tensor. Each partition consists of one or more adjacent
	blocks.

	Tensor indexes that are affected by this %symmetry element are
	specified using a mask.

	The number of partitions specifies how blocks will be grouped together.
	For the block %index space to be valid with the %symmetry element,
	the number of blocks along each affected dimension must be divisible
	by the number of partitions. Moreover, block sizes must correspond
	correctly from partition to partition. That is, if the partitions must
	have the same block structure.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_part : public symmetry_element_i<N, T> {
public:
	static const char *k_clazz; //!< Class name
	static const char *k_sym_type; //!< Symmetry type

private:
	block_index_space<N> m_bis; //!< Block %index space
	dimensions<N> m_bidims; //!< Block %index space dimensions
	dimensions<N> m_pdims; //!< Partition %index dimensions
	size_t *m_fmap; //!< Forward mapping
	size_t *m_rmap; //!< Reverse mapping
	bool *m_fsign; //!< Sign of the mappings
	mask<N> m_mask; //!< Mask of affected indexes

public:
	//!	\name Construction and destruction / assignment
	//@{

	/**	\brief Initializes the %symmetry element
		\param bis Block %index space.
		\param msk Mask of affected dimensions.
		\param npart Number of partitions along each dimension.
	 **/
	se_part(const block_index_space<N> &bis, const mask<N> &msk, size_t npart);

	/**	\brief Copy constructor
	 **/
	se_part(const se_part<N, T> &elem);

	/**	\brief Virtual destructor
	 **/
	virtual ~se_part();

	//@}



	//!	\name Manipulations
	//@{

	/**	\brief Adds a mapping between two partitions
		\param idx1 First partition %index.
		\param idx2 Second partition %index.
		\param sign Sign of the mapping (true positive, false negative)
	 **/
	void add_map(const index<N> &idx1, const index<N> &idx2, bool sign = true);

	//@}

	//! \name Access functions

	//@{

	/** \brief Returns the block index space for the partitions.
	 **/
	const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	/** \brief Returns the number of partitions.
	 **/
	size_t get_npart() const;

	/** \brief Returns the partition dimensions.
	 **/
	const dimensions<N> &get_pdims() const {
		return m_pdims;
	}

	/** \brief Returns the index to which idx is mapped directly
			(refers to forward mapping)
	 **/
	index<N> get_direct_map(const index<N> &idx) const;

	/** \brief Returns the sign of the map between the two indexes.
	 **/
	bool get_sign(const index<N> &from, const index<N> &to) const;

	/** \brief Check if there exists a map between two indexes
	 **/
	bool map_exists(const index<N> &from, const index<N> &to) const;

	void permute(const permutation<N> &perm);

	const mask<N> &get_mask() const {
		return m_mask;
	}

	//@}

	//!	\name Implementation of symmetry_element_i<N, T>
	//@{

	/**	\copydoc symmetry_element_i<N, T>::get_type()
	 **/
	virtual const char *get_type() const {
		return k_sym_type;
	}

	/**	\copydoc symmetry_element_i<N, T>::clone()
	 **/
	virtual symmetry_element_i<N, T> *clone() const {
		return new se_part<N, T>(*this);
	}

	/**	\copydoc symmetry_element_i<N, T>::is_valid_bis
	 **/
	virtual bool is_valid_bis(const block_index_space<N> &bis) const;

	/**	\copydoc symmetry_element_i<N, T>::is_allowed
	 **/
	virtual bool is_allowed(const index<N> &idx) const {

		return true;
	}

	/**	\copydoc symmetry_element_i<N, T>::apply(index<N>&)
	 **/
	virtual void apply(index<N> &idx) const;

	/**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
	 **/
	virtual void apply(index<N> &idx, transf<N, T> &tr) const;

	//@}

private:
	/**	\brief Builds the partition %dimensions, throws an exception
			if the arguments are invalid
	 **/
	static dimensions<N> make_pdims(const block_index_space<N> &bis,
		const mask<N> &msk, size_t npart);

	/** Adds the map a->b to the loop a is in.
	 **/
	void add_to_loop(size_t a, size_t b, bool sign);

	/**	\brief Returns true if the %index is a valid partition %index,
			false otherwise
	 **/
	bool is_valid_pidx(const index<N> &idx);

};


template<size_t N, typename T>
const char *se_part<N, T>::k_clazz = "se_part<N, T>";


template<size_t N, typename T>
const char *se_part<N, T>::k_sym_type = "part";


template<size_t N, typename T>
se_part<N, T>::se_part(const block_index_space<N> &bis, const mask<N> &msk,
	size_t npart) :

	m_bis(bis), m_bidims(m_bis.get_block_index_dims()),
	m_pdims(make_pdims(bis, msk, npart)), m_mask(msk),
	m_fmap(0), m_rmap(0), m_fsign(0) {

	static const char *method =
		"se_part(const block_index_space<N>&, const mask<N>&, size_t)";

	size_t mapsz = m_pdims.get_size();
	m_fmap = new size_t[mapsz];
	m_rmap = new size_t[mapsz];
	m_fsign = new bool[mapsz];
	for (size_t i = 0; i < mapsz; i++) {
		m_fmap[i] = m_rmap[i] = i;
		m_fsign[i] = true;
	}
}


template<size_t N, typename T>
se_part<N, T>::se_part(const se_part<N, T> &elem) :

	m_bis(elem.m_bis), m_bidims(elem.m_bidims), m_pdims(elem.m_pdims),
	m_mask(elem.m_mask), m_fmap(0), m_fsign(0) {

	size_t mapsz = m_pdims.get_size();
	m_fmap = new size_t[mapsz];
	m_rmap = new size_t[mapsz];
	m_fsign = new bool[mapsz];
	for (size_t i = 0; i < mapsz; i++) {
		m_fmap[i] = elem.m_fmap[i];
		m_rmap[i] = elem.m_rmap[i];
		m_fsign[i] = elem.m_fsign[i];
	}
}

template<size_t N, typename T>
se_part<N, T>::~se_part() {

	delete [] m_fmap; m_fmap = 0;
	delete [] m_rmap; m_rmap = 0;
	delete [] m_fsign; m_fsign = 0;
}


template<size_t N, typename T>
void se_part<N, T>::add_map(
		const index<N> &idx1, const index<N> &idx2, bool sign) {

	static const char *method =
		"add_map(const index<N>&, const index<N>&, bool)";

#ifdef LIBTENSOR_DEBUG
	if(!is_valid_pidx(idx1)) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"idx1");
	}
	if(!is_valid_pidx(idx2)) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"idx2");
	}
#endif // LIBTENSOR_DEBUG

	abs_index<N> aidx1(idx1, m_pdims), aidx2(idx2, m_pdims);
	size_t a = aidx1.get_abs_index(), b = aidx2.get_abs_index();

	if(a == b) return;
	if(a > b) std::swap(a, b);

	// check if b is in the same loop as a
	size_t ax = a, axf = m_fmap[ax];
	bool sx = true;
	while (ax < axf && ax < b) {
		sx = (sx == m_fsign[ax]);
		ax = axf; axf = m_fmap[ax];
	}
	if (ax == b) {
		if (sx != sign)
			throw bad_parameter(g_ns, k_clazz, method,
					__FILE__, __LINE__, "Mapping exists with different sign.");

		return;
	}

	size_t br = m_rmap[b], bf = m_fmap[b];
	bool sab(sign); // cur a -> cur b
	while (b != bf) {
		// remove b from its loop
		sx = m_fsign[b];
		m_fmap[br] = bf; m_rmap[bf] = br;
		m_fsign[br] = (m_fsign[br] == sx);

		// add it to the loop of a
		add_to_loop(a, b, sab);

		// go to next b
		a = b; b = bf; bf = m_fmap[b];
		sab = sx;
	}

	// add last b to loop
	add_to_loop(a, b, sab);
}

template<size_t N, typename T>
size_t se_part<N, T>::get_npart() const {

	for (size_t i = 0; i < N; i++) if (m_mask[i]) return m_pdims[i];

}

template<size_t N, typename T>
index<N> se_part<N, T>::get_direct_map(const index<N> &idx) const {

	abs_index<N> ai(idx, m_pdims);
	abs_index<N> aif(m_fmap[ai.get_abs_index()], m_pdims);
	return aif.get_index();
}

template<size_t N, typename T>
bool se_part<N, T>::get_sign(const index<N> &from, const index<N> &to) const {

	static const char *method = "get_sign(const index<N> &, const index<N> &)";
	size_t a = abs_index<N>(from, m_pdims).get_abs_index();
	size_t b = abs_index<N>(to, m_pdims).get_abs_index();

	if (a > b) std::swap(a, b);
	size_t x = m_fmap[a];
	bool sign = m_fsign[a];
	while (x != b && a < x) {
		sign = (sign == m_fsign[x]);
		x = m_fmap[x];

	}
	if (x <= a)
		throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, "No mapping.");

	return sign;
}

template<size_t N, typename T>
bool se_part<N, T>::map_exists(
		const index<N> &from, const index<N> &to) const {

	size_t a = abs_index<N>(from, m_pdims).get_abs_index();
	size_t b = abs_index<N>(to, m_pdims).get_abs_index();

	if (a > b) std::swap(a, b);

	size_t x = m_fmap[a];
	while (x != b && a < x) {
		x = m_fmap[x];
	}
	return (x == b);
}

template<size_t N, typename T>
bool se_part<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	return m_bis.equals(bis);
}

template<size_t N, typename T>
void se_part<N, T>::permute(const permutation<N> &perm) {

	if (perm.is_identity()) return;

	m_bis.permute(perm);
	m_bidims.permute(perm);

	sequence<N, size_t> seq(0);
	for (size_t i = 0; i < N; i++) seq[i] = i;
	perm.apply(seq);
	bool affects_map = false;
	for (size_t i = 0; i < N; i++) {
		if (m_mask[i] && seq[i] != i) { affects_map = true; break; }
	}

	if (affects_map) {
		dimensions<N> pdims(m_pdims);
		m_pdims.permute(perm);
		m_mask.permute(perm);

		size_t mapsz = m_pdims.get_size();
		size_t *fmap = m_fmap; m_fmap = new size_t[mapsz];
		size_t *rmap = m_rmap; m_rmap = new size_t[mapsz];
		bool *fsign = m_fsign; m_fsign = new bool[mapsz];
		for (size_t i = 0; i < mapsz; i++) {
			m_fmap[i] = m_rmap[i] = i;
			m_fsign[i] = true;
		}

		for (size_t i = 0; i < mapsz; i++) {

			if (fmap[i] <= i) continue;

			abs_index<N> aia(i, pdims);
			index<N> ia(aia.get_index());
			ia.permute(perm);
			abs_index<N> naia(ia, m_pdims);

			abs_index<N> aib(fmap[i], pdims);
			index<N> ib(aib.get_index());
			ib.permute(perm);
			abs_index<N> naib(ib, m_pdims);

			add_map(naia.get_index(), naib.get_index(), fsign[i]);
		}

		delete [] fmap; fmap = 0;
		delete [] rmap;	rmap = 0;
		delete [] fsign; fsign = 0;
	}
}

template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx) const {

	transf<N, T> tr;
	apply(idx, tr);
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx, transf<N, T> &tr) const {

	//	Determine partition index and offset within partition
	//
	index<N> pidx, poff;
	for(register size_t i = 0; i < N; i++) {
		if(m_pdims[i] == 1) {
			pidx[i] = 0;
			poff[i] = idx[i];
		} else {
			register size_t n = m_bidims[i] / m_pdims[i];
			pidx[i] = idx[i] / n;
			poff[i] = idx[i] % n;
		}
	}

	//	Map the partition index
	//
	abs_index<N> apidx(pidx, m_pdims);
	abs_index<N> apidx_mapped(m_fmap[apidx.get_abs_index()], m_pdims);
	pidx = apidx_mapped.get_index();

	//	Construct a mapped block index
	//
	for(register size_t i = 0; i < N; i++) {
		register size_t n = m_bidims[i] / m_pdims[i];
		idx[i] = pidx[i] * n + poff[i];
	}

	if (! m_fsign[apidx.get_abs_index()]) tr.scale(-1.0);
}


template<size_t N, typename T>
dimensions<N> se_part<N, T>::make_pdims(const block_index_space<N> &bis,
	const mask<N> &msk, size_t npart) {

	static const char *method = "make_pdims(const block_index_space<N>&, "
		"const mask<N>&, size_t)";

	if(npart < 2) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"npart");
	}

	index<N> i1, i2;
	size_t m = 0;
	for(register size_t i = 0; i < N; i++) {
		if(msk[i]) {
			i2[i] = npart - 1;
			m++;
		} else {
			i2[i] = 0;
		}
	}

	//	Make sure the partitioning is not trivial
	//
	if(m == 0) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"msk");
	}

	dimensions<N> pdims(index_range<N>(i1, i2));
	dimensions<N> bidims = bis.get_block_index_dims();

	//	Make sure the splits are identical for all partitions
	//
	for(size_t i = 0; i < N; i++) {

		size_t np = pdims[i];
		if(np == 1) continue;

		if(bidims[i] % np != 0) {
			throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, "bis");
		}

		size_t psz = bidims[i] / np;
		const split_points &pts = bis.get_splits(bis.get_type(i));
		size_t d = pts[psz - 1];
		for(size_t j = 0; j < psz; j++) {
			size_t pt0 = j == 0 ? 0 : pts[j - 1];
			for(size_t k = 1; k < np; k++) {
				if(pts[k * psz + j - 1] != pt0 + k * d) {
					throw bad_symmetry(g_ns, k_clazz,
						method, __FILE__, __LINE__,
						"bis");
				}
			}
		}
	}

	return pdims;
}


template<size_t N, typename T>
void se_part<N, T>::add_to_loop(size_t a, size_t b, bool sign) {

	if (a < b) {
		size_t af = m_fmap[a];
		while (af < b && a < af) {
			sign = (sign == m_fsign[a]);
			a = af; af = m_fmap[a];
		}
		m_fmap[a] = b; m_rmap[b] = a;
		m_fmap[b] = af; m_rmap[af] = b;
		m_fsign[b] = (sign == m_fsign[a]);
		m_fsign[a] = sign;
	}
	else {
		size_t ar = m_rmap[a];
		while (ar > b && ar < a) {
			sign = (m_fsign[ar] == sign);
			a = ar; ar = m_rmap[a];
		}
		m_fmap[ar] = b; m_rmap[b] = ar;
		m_fmap[b] = a; m_rmap[a] = b;
		m_fsign[ar] = (sign == m_fsign[a]);
		m_fsign[b] = sign;
	}
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_pidx(const index<N> &idx) {

	for(register size_t i = 0; i < N; i++)
		if(idx[i] >= m_pdims[i]) return false;
	return true;
}

} // namespace libtensor

#endif // LIBTENSOR_SE_PART_H

