#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "btensor_i.h"

namespace libtensor {

/**	\brief Container for a %tensor expression operand

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class labeled_btensor_expr_arg_tensor {
private:
	btensor_i<N, T> &m_bt;
	permutation<N> m_perm;
	T m_coeff;

public:
	labeled_btensor_expr_arg_tensor(btensor_i<N, T> &bt,
		permutation<N> &perm, T coeff) :
			m_bt(bt), m_perm(perm), m_coeff(coeff) { }
	void scale(T c) { m_coeff *= c; }
	btensor_i<N, T> &get_btensor();
	const permutation<N> &get_permutation() const;
	T get_coeff() const;
};

/**	\brief Container for a %tensor operation expression argument

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class labeled_btensor_expr_arg_oper {

};

template<size_t N, typename T>
inline btensor_i<N, T> &labeled_btensor_expr_arg_tensor<N, T>::get_btensor() {
	return m_bt;
}

template<size_t N, typename T>
inline const permutation<N>&
labeled_btensor_expr_arg_tensor<N, T>::get_permutation() const {
	return m_perm;
}

template<size_t N, typename T>
inline T labeled_btensor_expr_arg_tensor<N, T>::get_coeff() const {
	return m_coeff;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
