#ifndef LIBTENSOR_SE_PART_H
#define LIBTENSOR_SE_PART_H

#include <algorithm>
#include "../defs.h"
#include "../core/abs_index.h"
#include "../core/block_index_space.h"
#include "../core/mask.h"
#include "../core/symmetry_element_i.h"
#include "bad_symmetry.h"

namespace libtensor {


/**	\brief Symmetry between %tensor partitions
	\tparam N Symmetry cardinality (%tensor order).
	\tparam T Tensor element type.

	This %symmetry element established relationships between partitions
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
	size_t *m_fmap; //!< Partition map (forward)
	size_t *m_rmap; //!< Partition map (reverse)

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the %symmetry element
		\param bis Block %index space.
		\param msk Mask of affected dimensions.
		\param npart Number of partitions along each dimension.
	 **/
	se_part(const block_index_space<N> &bis, const mask<N> &msk,
		size_t npart);

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
	 **/
	void add_map(const index<N> &idx1, const index<N> &idx2);

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

	/**	\copydoc symmetry_element_i<N, T>::get_mask
	 **/
	virtual const mask<N> &get_mask() const {
		throw 1;
	}

	/**	\copydoc symmetry_element_i<N, T>::permute
	 **/
	virtual void permute(const permutation<N> &perm) {
		throw 1;
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
	static dimensions<N> make_pdims(const mask<N> &msk, size_t npart);
};


template<size_t N, typename T>
const char *se_part<N, T>::k_clazz = "se_part<N, T>";


template<size_t N, typename T>
const char *se_part<N, T>::k_sym_type = "part";


template<size_t N, typename T>
se_part<N, T>::se_part(const block_index_space<N> &bis, const mask<N> &msk,
	size_t npart) :

	m_bis(bis), m_bidims(m_bis.get_block_index_dims()),
	m_pdims(make_pdims(msk, npart)), m_fmap(0), m_rmap(0) {

	static const char *method =
		"se_part(const block_index_space<N>&, const mask<N>&, size_t)";

	//	Make sure the partitioning is not trivial
	//
	size_t m = 0;
	for(register size_t i = 0; i < N; i++) if(msk[i]) m++;
	if(m == 0) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"msk");
	}

	//	Make sure the splits are identical for all partitions
	//
	for(size_t i = 0; i < N; i++) {

		size_t np = m_pdims[i];
		if(np == 1) continue;

		if(m_bidims[i] % np != 0) {
			throw bad_symmetry(g_ns, k_clazz, method,
				__FILE__, __LINE__, "bis");
		}

		size_t psz = m_bidims[i] / np;
		const split_points &pts = m_bis.get_splits(m_bis.get_type(i));
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

	size_t mapsz = m_pdims.get_size();
	m_fmap = new size_t[mapsz];
	m_rmap = new size_t[mapsz];
	for(register size_t i = 0; i < mapsz; i++) m_fmap[i] = m_rmap[i] = i;
}


template<size_t N, typename T>
se_part<N, T>::se_part(const se_part<N, T> &elem) :

	m_bis(elem.m_bis), m_bidims(elem.m_bidims), m_pdims(elem.m_pdims) {

	size_t mapsz = m_pdims.get_size();
	m_fmap = new size_t[mapsz];
	m_rmap = new size_t[mapsz];
	for(register size_t i = 0; i < mapsz; i++) {
		m_fmap[i] = elem.m_fmap[i];
		m_rmap[i] = elem.m_rmap[i];
	}
}


template<size_t N, typename T>
se_part<N, T>::~se_part() {

	delete [] m_fmap;
	delete [] m_rmap;
}


template<size_t N, typename T>
void se_part<N, T>::add_map(const index<N> &idx1, const index<N> &idx2) {

	abs_index<N> aidx1(idx1, m_pdims), aidx2(idx2, m_pdims);
	size_t a = aidx1.get_abs_index(), b = aidx2.get_abs_index();

	if(a == b) return;
	if(a > b) std::swap(a, b);

	size_t af = m_fmap[a], bf = m_fmap[b], ar = m_rmap[a], br = m_rmap[b];

	if(af == a && bf == b) {
		m_fmap[a] = b; m_rmap[b] = a;
		m_fmap[b] = a; m_rmap[a] = b;
	} else if(af != a && bf == b) {
		m_fmap[a] = b; m_rmap[b] = a;
		m_fmap[b] = af; m_rmap[af] = b;
	} else if(af == a && bf != b) {
		m_fmap[a] = b; m_rmap[b] = a;
		m_fmap[br] = a; m_rmap[a] = br;
	} else {
		m_fmap[a] = b; m_rmap[b] = a;
		m_fmap[ar] = bf; m_rmap[bf] = ar;
		m_fmap[b] = af; m_rmap[af] = b;
		m_fmap[br] = a; m_rmap[a] = br;
	}
}


template<size_t N, typename T>
bool se_part<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	return m_bis.equals(bis);
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx) const {

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
}


template<size_t N, typename T>
void se_part<N, T>::apply(index<N> &idx, transf<N, T> &tr) const {

	apply(idx);
}


template<size_t N, typename T>
dimensions<N> se_part<N, T>::make_pdims(const mask<N> &msk, size_t npart) {

	static const char *method = "make_pdims(const mask<N>&, size_t)";

	if(npart < 2) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"npart");
	}

	index<N> i1, i2;
	for(register size_t i = 0; i < N; i++) {
		if(msk[i]) i2[i] = npart - 1;
		else i2[i] = 0;
	}
	return dimensions<N>(index_range<N>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_SE_PART_H