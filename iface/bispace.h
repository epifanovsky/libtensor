#ifndef LIBTENSOR_BISPACE_H
#define	LIBTENSOR_BISPACE_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "bispace_i.h"
#include "bispace_expr.h"
#include "core/dimensions.h"

namespace libtensor {

/**	\brief Block %index space defined by an expression
	\tparam N Order of the block %index space.

	\ingroup libtensor
 **/
template<size_t N>
class bispace : public bispace_i<N> {
private:
	rc_ptr<bispace_expr_i<N> > m_sym_expr; //!< Symmetry-defining expression
	dimensions<N> m_dims; //!< Correctly permuted dimensions
	block_index_space<N> m_bis; //!< Block %index space
	permutation<N> m_perm; //!< Permutation of indexes
	size_t m_inv_idx[N]; //!< Permuted index pointers for operator[]

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Constructs the block %index space using an expression
	 **/
	template<typename SymExprT>
	bispace(const bispace_exprr<N, SymExprT> &e_sym) throw(exception);

	/**	\brief Constructs the block %index space using expressions
		\param e_order Expession defining the order of components.
		\param e_sym Expression defining the symmetry of components.
		\throw exception If the expressions are invalid.

		This constructor inspects the order of %index space components
		in both parameters. The constructed space will use \e e_sym for
		symmetry purposes, but will permute indexes as defined in
		\e e_order. Both expressions must consist of identical
		components, otherwise they are considered invalid.
	 **/
	template<typename OrderExprT, typename SymExprT>
	bispace(const bispace_exprr<N, OrderExprT> &e_order,
		const bispace_exprr<N, SymExprT> &e_sym) throw(exception);

	/**	\brief Virtual destructor
	 **/
	virtual ~bispace();

	//@}

private:
	/**	\brief Private cloning constructor
	 **/
	bispace(const bispace<N> &other);

	void make_inv_idx();

public:
	//!	\name Implementation of libtensor::bispace_i<N>
	//@{
	virtual rc_ptr< bispace_i<N> > clone() const;
	virtual const block_index_space<N> &get_bis() const;
	//@}

	//!	\name Implementation of libtensor::block_index_space_i<N>
	//@{
	virtual const dimensions<N> &get_dims() const;
	//@}

	bool equals(const bispace<N> &bis) const {
		return m_bis.equals(bis.m_bis);
	}

	const bispace<1> &at(size_t) const;
	const bispace<1> &operator[](size_t i) const;
};

/**	\brief Specialized version for one-dimensional block %index spaces

	\ingroup libtensor
 **/
template<>
class bispace<1> : public bispace_i<1> {
private:
	block_index_space<1> m_bis; //!< Block %index space
	std::list<size_t> m_splits; //!< Split points
	size_t m_dim;

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the block %index space with a given dimension
	 **/
	bispace(size_t dim);

	/**	\brief Copy constructor
	 **/
	bispace(const bispace<1> &other);

	/**	\brief Virtual destructor
	 **/
	virtual ~bispace();

	//@}

	bool equals(const bispace<1> &other) const;

	/**	\brief Splits the space at a given position
		\param pos Position at which the space is to be split
	 **/
	bispace<1> &split(size_t pos) throw(exception);

	template<size_t N>
	void transfer_splits(block_index_space<N> &bis, size_t dim) const;

	//!	\name Implementation of bispace_i<1>
	//@{

	virtual rc_ptr<bispace_i < 1 > > clone() const;
	virtual const block_index_space<1> &get_bis() const;

	//@}

	//!	\name Implementation of block_index_space_i<1>
	//@{

	virtual const dimensions < 1 > &get_dims() const;

	//@}

	const bispace < 1 > & operator[](size_t i) const;

private:
	/**	\brief Private constructor for cloning
	 **/
	bispace(const dimensions < 1 > &dims);

	static dimensions < 1 > make_dims(size_t sz);
};

template<size_t N>
template<typename SymExprT>
bispace<N>::bispace(const bispace_exprr<N, SymExprT> &e_sym) throw(exception)
: m_sym_expr(e_sym.clone()), m_dims(e_sym.get_dims()), m_bis(m_dims) {

	make_inv_idx();
	for(size_t i = 0; i < N; i++) at(i).transfer_splits(m_bis, i);
}

template<size_t N>
template<typename OrderExprT, typename SymExprT>
bispace<N>::bispace(const bispace_exprr<N, OrderExprT> &e_order,
	const bispace_exprr<N, SymExprT> &e_sym) throw(exception)
: m_sym_expr(e_sym.clone()), m_dims(e_sym.get_dims()), m_bis(m_dims) {

	const bispace < 1 > *seq_order[N], *seq_sym[N];
	for(size_t i = 0; i < N; i++) {
		seq_order[i] = &(e_order[i]);
		seq_sym[i] = &(e_sym[i]);
	}
	// check uniqueness here
	bool done = false;
	while(!done) {
		permutation<N> p;
		size_t i, j;
		done = true;
		for(i = 0; i < N; i++) if(seq_sym[i] != seq_order[i]) break;
		if(i < N) {
			for(j = 0; j < N; j++) {
				if(seq_sym[i] == seq_order[j]) break;
			}
			if(j == N) {
				// exception
			}
			p.permute(i, j);
			m_perm.permute(i, j);
			p.apply(N, seq_sym);
			done = false;
		}
	}
	m_dims.permute(m_perm);
	m_bis.permute(m_perm);
	make_inv_idx();
	permutation<N> inv_perm(m_perm); inv_perm.invert();
	inv_perm.apply(N, m_inv_idx);
	for(size_t i = 0; i < N; i++) at(i).transfer_splits(m_bis, i);

}

template<size_t N>
bispace<N>::bispace(const bispace<N> &other)
: m_sym_expr(other.m_sym_expr->clone()), m_perm(other.m_perm),
	m_dims(other.m_dims), m_bis(other.m_bis) {

	make_inv_idx();
}

template<size_t N>
bispace<N>::~bispace() {

}

template<size_t N>
void bispace<N>::make_inv_idx() {

	for(size_t i = 0; i < N; i++) m_inv_idx[i] = i;
}

template<size_t N>
rc_ptr< bispace_i<N> > bispace<N>::clone() const {

	return rc_ptr< bispace_i<N> >(new bispace<N > (*this));
}

template<size_t N>
inline const dimensions<N> &bispace<N>::get_dims() const {

	return m_dims;
}

template<size_t N>
inline const block_index_space<N> &bispace<N>::get_bis() const {

	return m_bis;
}

template<size_t N>
inline const bispace<1> &bispace<N>::at(size_t i) const {

	return (*m_sym_expr)[m_inv_idx[i]];
}

template<size_t N>
inline const bispace<1> &bispace<N>::operator[](size_t i) const {

	return at(i);
}

inline bispace<1>::bispace(size_t dim)
: m_dim(dim), m_bis(make_dims(dim)) {

}

inline bispace<1>::bispace(const bispace<1> &other)
: m_dim(other.m_dim), m_bis(other.m_bis), m_splits(other.m_splits) {

}

inline bispace<1>::~bispace() {

}

inline bool bispace<1>::equals(const bispace<1> &other) const {

	if(this == &other) return true;
	if(m_dim != other.m_dim) return false;
	if(m_splits.size() != other.m_splits.size()) return false;
	std::list<size_t>::const_iterator i = m_splits.begin();
	std::list<size_t>::const_iterator j = other.m_splits.begin();
	for(; i != m_splits.end(); i++, j++) {
		if(*i != *j) return false;
	}
	return true;
}


inline bispace<1> &bispace<1>::split(size_t pos) throw(exception) {

	std::list<size_t>::iterator i = m_splits.begin();
	while(i != m_splits.end() && *i <= pos) i++;
	mask<1> msk; msk[0] = true;
	if(i == m_splits.end()) {
		m_splits.push_back(pos);
		m_bis.split(msk, pos);
	} else if(*i > pos) {
		m_splits.insert(i, pos);
		m_bis.split(msk, pos);
	}
	return *this;
}

template<size_t N>
inline void bispace<1>::transfer_splits(block_index_space<N> &bis, size_t dim)
	const {

	typename std::list<size_t>::const_iterator i = m_splits.begin();
	mask<N> msk; msk[dim] = true;
	while(i != m_splits.end()) {
		bis.split(msk, *i);
		i++;
	}
}

inline rc_ptr< bispace_i<1> > bispace<1>::clone() const {

	return rc_ptr< bispace_i<1> >(new bispace<1>(*this));
}

inline const dimensions<1> &bispace<1>::get_dims() const {

	return m_bis.get_dims();
}

inline const block_index_space<1> &bispace<1>::get_bis() const {

	return m_bis;
}

inline const bispace < 1 > &bispace < 1 > ::operator[](size_t i) const {
	if(i != 0) {
		// throw exception
	}
	return *this;
}

inline dimensions<1> bispace<1>::make_dims(size_t sz) {

	index<1> i1, i2; i2[0] = sz - 1;
	return dimensions<1>(index_range<1>(i1, i2));
}

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_H

