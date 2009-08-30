#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H

#include "defs.h"
#include "exception.h"
#include "core/permutation.h"
#include "iface/btensor_i.h"

namespace libtensor {
namespace labeled_btensor_expr {

struct tensor_tag { };
struct oper_tag { };

/**	\brief Generic container for an expression argument

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Tag>
class arg {
};

/**	\brief Container for a %tensor expression operand

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class arg<N, T, tensor_tag> {
private:
	btensor_i<N, T> &m_bt;
	permutation<N> m_perm;
	T m_coeff;

public:
	arg(btensor_i<N, T> &bt, const permutation<N> &perm, T coeff)
	: m_bt(bt), m_perm(perm), m_coeff(coeff) {
	}

	void scale(T c) {
		m_coeff *= c;
	}

	btensor_i<N, T> &get_btensor() {
		return m_bt;
	}

	const permutation<N> &get_perm() const {
		return m_perm;
	}

	T get_coeff() const {
		return m_coeff;
	}
};

/**	\brief Container for a %tensor operation expression argument

	\ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class arg<N, T, oper_tag> {
};

/**	\brief Container for a %tensor operation expression argument
		(specialized for double)

	\ingroup libtensor_btensor_expr
 **/
template<size_t N>
class arg<N, double, oper_tag> {
private:
	btod_additive<N> &m_op;
	double m_coeff;

public:
	arg(btod_additive<N> &op, double coeff) : m_op(op), m_coeff(coeff) {
	}

	void scale(double c) {
		m_coeff *= c;
	}

	btod_additive<N> &get_operation() {
		return m_op;
	}

	double get_coeff() const {
		return m_coeff;
	}
};

} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_ARG_H
