#ifndef LIBTENSOR_BISPACE_H
#define	LIBTENSOR_BISPACE_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../core/dimensions.h"
#include "../core/mask.h"
#include "../core/sequence.h"
#include "../core/split_points.h"
#include "bispace_expr/expr.h"
#include "bispace_expr/ident.h"
#include "bispace_expr/asym.h"
#include "bispace_expr/sym.h"
#include "bispace_expr/operator_or.h"
#include "bispace_expr/operator_and.h"

namespace libtensor {


/**	\brief Block %index space defined using an expression
	\tparam N Order of the block %index space.

	This class, along with expressions, provides a user-friendly mechanism
	of creating block %index spaces. The construction of the spaces starts
	with defining each dimension and block splitting points.

	\code
	bispace<1> i(10); // This creates space i of 10 elements
	bispace<1> a(20); // Space a consists of 20 elements

	i.split(5); // Splits space i into two blocks, 5 elements each
	a.split(10).split(15); // Splits a into 3 blocks: [10, 5, 5]

	// Create copies of i and a, including splittings
	bispace<1> j(i), b(a);
	\endcode

	One-dimensional spaces can be used on their own (for example, with
	one-dimensional tensors -- vectors) or combined to form
	multi-dimensional spaces. Two operators are provided to synthesize
	spaces: operator& for symmetry and operator| for asymmetry. Subspaces
	connected using operator& are marked as suitable for permutational
	symmetry, and therefore must be compatible. Compatible subspaces
	contain the same number of elements and are split into blocks using
	the same pattern. If this condition is not observed, an attempt to
	create symmetry will cause an exception. Operator| can be used with
	any subspaces, compatible or otherwise. No symmetry relations are
	allowed among asymmetric subspaces.

	\code
	// Create symmetric subspaces ij and ab
	bispace<2> ij(i&j), ab(a&b);

	// Create space ijab
	bispace<4> ijab1(ij|ab);

	// The same can be done using the original i, j, a, b
	bispace<4> ijab2(i&j|a&b);

	// Spaces ijab1 and ijab2 are equivalent
	\endcode

	If a single expression cannot be used to specify both symmetry and
	the correct order of subspaces, two expressions are specified: one
	for the order of indexes and the other for symmetry relations.

	\code
	// Symmetry between i and j, a and b, but the final order is iajb
	bispace<4> iajb(i|a|j|b, i&j|a&b);
	\endcode

	Whenever two expressions are used to create a block %index space, they
	must consist of the same unique components, otherwise the permutation
	cannot be determined. In the case of a single expression, repetitions
	are allowed (although discouraged for readability purposes).

	\code
	bispace<4> ijab1(i&j|a&b); // OK
	bispace<4> ijab2(i&i|a&a); // OK, same as above
	bispace<4> iajb1(i|a|j|b, i&j|a&b); // OK
	bispace<4> iajb2(i|a|i|a, i&i|a&a); // Exception!
	bispace<4> iajb3(i|a|j|b, i&i|a&a); // Exception!
	\endcode


	\ingroup libtensor_iface
 **/
template<size_t N>
class bispace {
public:
	static const char *k_clazz; //!< Class name

private:
	sequence< N, bispace<1>* > m_subspaces; //!< Subspaces
	block_index_space<N> m_bis; //!< Block %index space
	std::list< mask<N> > m_masks; //!< Symmetry masks

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructs the block %index space using an expression
	 **/
	template<typename C>
	bispace(const bispace_expr::expr<N, C> &esym);

	/**	\brief Constructs the block %index space using two expressions:
			for symmetry and for the order of indexes
		\param eord Expession defining the order of subspaces.
		\param esym Expression defining the symmetry of subspaces.
		\throw expr_exception If the expressions are invalid.

		This constructor inspects the order of %index space components
		in both parameters. The constructed space will use \e e_sym for
		symmetry purposes, but will permute indexes as defined in
		\e e_order. Both expressions must consist of identical
		components, otherwise they are considered invalid.
	 **/
	template<typename C1, typename C2>
	bispace(const bispace_expr::expr<N, C1> &eord,
		const bispace_expr::expr<N, C2> &esym);

	/**	\brief Copy constructor
	 **/
	bispace(const bispace<N> &bis);

	/**	\brief Destructor
	 **/
	~bispace();

	//@}


	//!	\name Block %index space structure and comparison
	//@{

	/**	\brief Returns true if two libtensor::bispace<N> objects are
			equal (set up the same block %index space)
	 **/
	bool equals(const bispace<N> &other) const {
		return m_bis.equals(other.m_bis);
	}

	/**	\brief Returns the libtensor::block_index_space<N> object
	 **/
	const block_index_space<N> &get_bis() const {
		return m_bis;
	}

	/**	\brief Returns a one-dimensional subspace at a given %index
			position
		\throw out_of_bounds If the %index position is i >= N.
	 **/
	const bispace<1> &at(size_t i) const {
		return *m_subspaces[i];
	}

	/**	\brief Returns a %mask that indicates symmetry relationships
			of one %index with other indexes
	 **/
	const mask<N> &get_sym_mask(size_t i) const;

	//@}

private:
	/**	\brief Creates a libtensor::dimensions<N> object from an
			expression
	 **/
	template<typename C>
	static dimensions<N> mk_dims(const bispace_expr::expr<N, C> &expr);

	/**	\brief Transfers splits and populates the mask list
	 **/
	template<typename C>
	void splits_and_masks(const bispace_expr::expr<N, C> &esym);

};


/**	\brief One-dimensional block %index space (special case)

	This class specializes the libtensor::bispace<N> template. Refer to it
	for more information.

	\ingroup libtensor_iface
 **/
template<>
class bispace<1> {
private:
	size_t m_dim; //!< Number of elements
	block_index_space<1> m_bis; //!< Block %index space
	mask<1> m_msk; //!< Symmetry mask

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the block %index space with a given dimension
		\param dim Number of elements in this space.
	 **/
	bispace(size_t dim);

	/**	\brief Copy constructor
	 **/
	bispace(const bispace<1> &other);

	/**	\brief Destructor
	 **/
	~bispace();

	//@}


	//!	\name Block %indes space manipulations
	//@{

	/**	\brief Splits the space at a given position
		\param pos Position at which the space is to be split.
	 **/
	bispace<1> &split(size_t pos);

	/**	\brief Copies split points from this one-dimensional space to
			a multi-dimensional block %index space using a %mask
	 **/
	template<size_t N>
	void copy_splits(block_index_space<N> &bis, const mask<N> &msk) const;

	//@}


	//!	\name Block %index space structure and comparison
	//@{

	/**	\brief Returns true if two single-dimension block %index
			spaces are identical
	 **/
	bool equals(const bispace<1> &other) const;

	/**	\brief Returns the number of elements along this dimension
	 **/
	size_t get_dim() const {
		return m_dim;
	}

	/**	\brief Returns the libtensor::block_index_space<N> object
	 **/
	const block_index_space<1> &get_bis() const {
		return m_bis;
	}

	/**	\brief Returns this one-dimensional subspace
		\throw out_of_bounds If the %index position is i != 0.
	 **/
	const bispace<1> &at(size_t i) const;

	/**	\brief Returns a one-dimensional %mask set to one
	 **/
	const mask<1> &get_sym_mask(size_t i) const;

	//@}

private:
	/**	\brief Constructs a dimensions object using the number of
			elemens
	 **/
	static dimensions<1> mk_dims(size_t dim);

};


template<size_t N>
const char *bispace<N>::k_clazz = "bispace<N>";


template<size_t N> template<typename C>
bispace<N>::bispace(const bispace_expr::expr<N, C> &esym) :
	m_subspaces(NULL), m_bis(mk_dims(esym)) {

	for(size_t i = 0; i < N; i++) {
		m_subspaces[i] = new bispace<1>(esym.at(i));
	}

	splits_and_masks(esym);
}


template<size_t N> template<typename C1, typename C2>
bispace<N>::bispace(const bispace_expr::expr<N, C1> &eord,
	const bispace_expr::expr<N, C2> &esym) :
		m_subspaces(NULL), m_bis(mk_dims(esym)) {

	for(size_t i = 0; i < N; i++) {
		m_subspaces[i] = new bispace<1>(esym.at(i));
	}

	splits_and_masks(esym);

	permutation<N> perm;
	eord.build_permutation(esym, perm);
	m_subspaces.permute(perm);
	m_bis.permute(perm);
	typename std::list< mask<N> >::iterator imsk = m_masks.begin();
	for(; imsk != m_masks.end(); imsk++) imsk->permute(perm);
}


template<size_t N>
bispace<N>::bispace(const bispace<N> &other) :
	m_subspaces(NULL), m_bis(other.m_bis), m_masks(other.m_masks) {

	for(size_t i = 0; i < N; i++) {
		m_subspaces[i] = new bispace<1>(*other.m_subspaces[i]);
	}
}


template<size_t N>
bispace<N>::~bispace() {

	for(size_t i = 0; i < N; i++) {
		delete m_subspaces[i];
		m_subspaces[i] = NULL;
	}
}


template<size_t N>
const mask<N> &bispace<N>::get_sym_mask(size_t i) const {

	static const char *method = "get_sym_mask(size_t)";

	typename std::list< mask<N> >::const_iterator imsk = m_masks.begin();
	for(; imsk != m_masks.end(); imsk++) {
		if(imsk->at(i)) return *imsk;
	}
	throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Symmetry mask cannot be found.");
}


template<size_t N> template<typename C>
dimensions<N> bispace<N>::mk_dims(const bispace_expr::expr<N, C> &expr) {

	index<N> i1, i2;
	for(size_t i = 0; i < N; i++) {
		i2[i] = expr.at(i).get_dim() - 1;
	}
	dimensions<N> dims(index_range<N>(i1, i2));
	return dims;
}


template<size_t N> template<typename C>
void bispace<N>::splits_and_masks(const bispace_expr::expr<N, C> &expr) {

	static const char *method =
		"splits_and_masks(const bispace_expr::expr<N, C>&)";

	mask<N> totmsk;
	size_t i = 0;
	while(true) {
		while(i < N && totmsk[i]) i++;
		if(i == N) break;
		mask<N> msk;
		expr.mark_sym(i, msk);
#ifdef LIBTENSOR_DEBUG
		if(!msk[i]) {
			throw expr_exception(g_ns, k_clazz, method,
				__FILE__, __LINE__, "Incorrect symmetry mask.");
		}
#endif // LIBTENSOR_DEBUG
		m_masks.push_back(msk);
		expr.at(i).copy_splits(m_bis, msk);
		totmsk |= msk;
	}
}


inline bispace<1>::bispace(size_t dim) : m_dim(dim), m_bis(mk_dims(dim)) {

	m_msk[0] = true;
}


inline bispace<1>::bispace(const bispace<1> &other) :
	m_dim(other.m_dim),
	m_bis(other.m_bis),
	m_msk(other.m_msk) {

}


inline bispace<1>::~bispace() {

}


inline bispace<1> &bispace<1>::split(size_t pos) {

	mask<1> msk; msk[0] = true;
	m_bis.split(msk, pos);
	return *this;
}


template<size_t N>
inline void bispace<1>::copy_splits(
	block_index_space<N> &bis, const mask<N> &msk) const {

	const split_points &splits = m_bis.get_splits(m_bis.get_type(0));
	size_t n = splits.get_num_points();
	for(size_t i = 0; i < n; i++) {
		bis.split(msk, splits[i]);
	}
}


inline bool bispace<1>::equals(const bispace<1> &other) const {

	if(this == &other) return true;
	return m_bis.equals(other.m_bis);
}


inline const bispace<1> &bispace<1>::at(size_t i) const {

	if(i != 0) {
		throw out_of_bounds(g_ns, "bispace<1>", "at(size_t)",
			__FILE__, __LINE__,
			"Dimension index is out of bounds.");
	}
	return *this;
}


inline const mask<1> &bispace<1>::get_sym_mask(size_t i) const {

	if(i != 0) {
		throw out_of_bounds(g_ns, "bispace<1>", "get_sym_mask(size_t)",
			__FILE__, __LINE__,
			"Dimension index is out of bounds.");
	}
	return m_msk;
}


inline dimensions<1> bispace<1>::mk_dims(size_t dim) {

	if(dim == 0) {
		throw bad_parameter(g_ns, "bispace<1>", "mk_dims(size_t)",
			__FILE__, __LINE__,
			"Positive number of elements is required.");
	}
	index<1> i1, i2;
	i2[0] = dim - 1;
	return dimensions<1>(index_range<1>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_BISPACE_H

