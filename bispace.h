#ifndef LIBTENSOR_BISPACE_H
#define	LIBTENSOR_BISPACE_H

#include <list>
#include "defs.h"
#include "exception.h"
#include "bispace_i.h"
#include "bispace_expr.h"
#include "dimensions.h"

namespace libtensor {

/**	\brief Block %index space defined by an expression
	\tparam N Order of the block %index space.
	\tparam SymExprT Symmetry-defining expression type.

	\ingroup libtensor
 **/
template<size_t N, typename SymExprT=void>
class bispace : public bispace_i<N> {
	typedef SymExprT sym_expr_t; //!< Symmetry-defining expression type

private:
	sym_expr_t m_sym_expr; //!< Symmetry-defining expression
	permutation<N> m_perm; //!< Permutation of indexes

public:
	//!	\name Construction and destruction
	//@{

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
	template<typename OrderExprT>
	bispace(const bispace_expr<N, OrderExprT> &e_order,
		const bispace_expr<N, SymExprT> &e_sym) throw(exception);

	/**	\brief Virtual destructor
	 **/
	virtual ~bispace();

	//@}

private:
	/**	\brief Private cloning constructor
		\param e_sym Expression defining the symmetry of components
		\param perm Permutation of components
	 **/
	bispace(const bispace_expr<N, SymExprT> &e_sym,
		const permutation<N> &perm);

public:
	//!	\name Implementation of libtensor::bispace_i<N>
	//@{
	virtual rc_ptr< bispace_i<N> > clone() const;
	//@}
};

/**	\brief Special version for one-dimensional block %index spaces

	\ingroup libtensor
 **/
template<typename SymT>
class bispace < 1, SymT> : public bispace_i < 1 > {
private:
	dimensions < 1 > m_dims; //!< Space %dimensions

public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Creates the block %index space with a given dimension
	 **/
	bispace(size_t dim);

	/**	\brief Copy constructor
	 **/
	bispace(const bispace<1,SymT> &other);

	/**	\brief Virtual destructor
	 **/
	virtual ~bispace();

	//@}

	/**	\brief Splits the space at a given position
	 **/
	bispace<1,SymT> &split(size_t pos) throw(exception);

	//!	\name Implementation of bispace_i<1>
	//@{

	virtual rc_ptr<bispace_i < 1 > > clone() const;

	//@}

	//!	\name Implementation of ispace_i<1>
	//@{

	virtual const dimensions < 1 > &dims() const;

	//@}

private:
	/**	\brief Private constructor for cloning
	 **/
	bispace(const dimensions < 1 > &dims);

	static dimensions < 1 > make_dims(size_t sz);
};

template<size_t N, typename SymExprT> template<typename OrderExprT>
bispace<N, SymExprT>::bispace(const bispace_expr<N, OrderExprT> &e_order,
	const bispace_expr<N, SymExprT> &e_sym) throw(exception) : m_sym_expr(e_sym) {

}

template<size_t N, typename SymExprT>
rc_ptr< bispace_i<N> > bispace<N, SymExprT>::clone() const {
	return rc_ptr< bispace_i<N> >(
		new bispace<N, SymExprT > (m_sym_expr, m_perm));
}

template<typename SymT>
inline bispace < 1, SymT>::bispace(size_t dim) : m_dims(make_dims(dim)) {
}

template<typename SymT>
inline bispace<1,SymT>::bispace(const bispace<1,SymT> &other) :
m_dims(other.m_dims) {

}

template<typename SymT>
inline bispace<1,SymT>::bispace(const dimensions < 1 > &dims) : m_dims(dims) {
}

template<typename SymT>
inline bispace < 1, SymT>::~bispace() {
}

template<typename SymT>
bispace<1,SymT> &bispace<1,SymT>::split(size_t pos) throw(exception) {
	return *this;
}

template<typename SymT>
inline rc_ptr<bispace_i < 1 > > bispace<1,SymT>::clone() const {
	return rc_ptr<bispace_i < 1 > >(new bispace<1,SymT>(m_dims));
}

template<typename SymT>
inline const dimensions < 1 > &bispace < 1, SymT>::dims() const {
	return m_dims;
}

template<typename SymT>
dimensions < 1 > bispace < 1, SymT>::make_dims(size_t sz) {
	index < 1 > i1, i2;
	i2[0] = sz - 1;
	index_range < 1 > ir(i1, i2);
	return dimensions < 1 > (ir);
}

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_H

