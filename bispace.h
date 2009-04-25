#ifndef LIBTENSOR_BISPACE_H
#define	LIBTENSOR_BISPACE_H

#include "defs.h"
#include "exception.h"
#include "bispace_i.h"
#include "bispace_expr.h"

namespace libtensor {

/**	\brief Block %index space defined by an expression
	\tparam N Order of the block %index space.
	\tparam SymExprT Symmetry-defining expression type.

	\ingroup libtensor
 **/
template<size_t N, typename SymExprT>
class bispace {
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
		const bispace_expr<N, SymExprT> &e_sym) throw (exception);

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

template<size_t N, typename SymExprT> template<typename OrderExprT>
bispace<N, SymExprT>::bispace(const bispace_expr<N, OrderExprT> &e_order,
	const bispace_expr<N, SymExprT> &e_sym) throw(exception) : m_sym_expr(e_sym) {

}

template<size_t N, typename SymExprT>
rc_ptr< bispace_i<N> > bispace<N, SymExprT>::clone() const {
	return rc_ptr< bispace_i<N> >(
		new bispace<N, SymExprT > (m_sym_expr, m_perm));
}

} // namespace libtensor

#endif // LIBTENSOR_BISPACE_H

