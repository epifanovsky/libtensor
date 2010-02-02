#ifndef LIBTENSOR_SE_PERM_H
#define LIBTENSOR_SE_PERM_H

#include "../defs.h"
#include "../core/mask.h"
#include "../core/permutation.h"
#include "../core/sequence.h"
#include "../core/symmetry_element_i.h"
#include "../core/transf.h"
#include "bad_symmetry.h"

namespace libtensor {


/**	\brief Permutational %symmetry element
	\tparam N Tensor order.
	\tparam T Tensor element type.

	Permutation %symmetry elements establish relationships among block
	%tensor blocks based on permuting the blocks' multi-indexes.
	An appropriate transformation needs to be applied to the elements in
	the block as well.

	The element is initialized with a %permutation of %tensor indexes,
	which also specifies a transformation of %tensor elements. The
	symmetric/anti-symmetric flag yields the scalar coefficient in
	the transformation. The permutation and the flag must agree: when
	the block transformation is applied onto itself multiple times such
	that the original permutation is recovered, the scalar coefficient
	must turn unity. The agreement is tested upon the creation of the
	element. If it is not satisfied, an exception is thrown.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class se_perm : public symmetry_element_i<N, T> {
public:
	static const char *k_clazz; //!< Class name
	static const char *k_sym_type; //!< Symmetry type

private:
	permutation<N> m_perm; //!< Permutation
	bool m_symm; //!< Symmetric/anti-symmetric
	transf<N, T> m_transf; //!< Block transformation
	mask<N> m_mask; //!< Mask of affected indexes

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Initializes the %symmetry element
		\param perm Permutation.
		\param symm Symmetric/anti-symmetric.
		\throw bad_symmetry If the permutation and the flag are
			inconsistent.
	 **/
	se_perm(const permutation<N> &perm, bool symm);

	/**	\brief Copy constructor
	 **/
	se_perm(const se_perm<N, T> &elem);

	/**	\brief Virtual destructor
	 **/
	virtual ~se_perm() { }

	//@}


	//!	\name Permutational %symmetry
	//@{

	const permutation<N> &get_perm() const {
		return m_perm;
	}

	bool is_symm() const {
		return m_symm;
	}

	const transf<N, T> &get_transf() const {
		return m_transf;
	}


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
		return new se_perm<N, T>(*this);
	}

	/**	\copydoc symmetry_element_i<N, T>::get_mask
	 **/
	virtual const mask<N> &get_mask() const {
		return m_mask;
	}

	/**	\copydoc symmetry_element_i<N, T>::permute
	 **/
	virtual void permute(const permutation<N> &perm) {

/*
		m_transf.permute(perm);
		m_mask.permute(perm);
*/
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
	virtual void apply(index<N> &idx) const {

		idx.permute(m_transf.get_perm());
	}

	/**	\copydoc symmetry_element_i<N, T>::apply(
			index<N>&, transf<N, T>&)
	 **/
	virtual void apply(index<N> &idx, transf<N, T> &tr) const {

		idx.permute(m_transf.get_perm());
		tr.transform(m_transf);
	}

	//@}
};


template<size_t N, typename T>
const char *se_perm<N, T>::k_clazz = "se_perm<N, T>";


template<size_t N, typename T>
const char *se_perm<N, T>::k_sym_type = "perm";


template<size_t N, typename T>
se_perm<N, T>::se_perm(const permutation<N> &perm, bool symm) :
	m_perm(perm), m_symm(symm) {

	static const char *method = "se_perm(const permutation<N>&, bool)";

	if(perm.is_identity()) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"perm.is_identity()");
	}

	size_t n = 0;
	permutation<N> p(m_perm);
	do {
		p.permute(m_perm); n++;
	} while(!p.is_identity());

	if(n % 2 == 0 && !m_symm) {
		throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
			"perm");
	}

	m_transf.permute(m_perm);
	if(!m_symm) m_transf.scale(-1);

	size_t seq[N];
	for(size_t i = 0; i < N; i++) seq[i] = i;
	perm.apply(seq);
	for(size_t i = 0; i < N; i++) if(seq[i] != i) m_mask[i] = true;
}


template<size_t N, typename T>
se_perm<N, T>::se_perm(const se_perm<N, T> &elem) :
	m_perm(elem.m_perm), m_symm(elem.m_symm), m_transf(elem.m_transf),
	m_mask(elem.m_mask) {

}


template<size_t N, typename T>
bool se_perm<N, T>::is_valid_bis(const block_index_space<N> &bis) const {

	block_index_space<N> bis2(bis);
	bis2.permute(m_perm);
	return bis2.equals(bis);
}


} // namespace libtensor

#endif // LIBTENSOR_SE_PERM_H
