#ifndef LIBTENSOR_BISPACE_EXPR_ASYM_H
#define LIBTENSOR_BISPACE_EXPR_ASYM_H

#include "expr.h"

namespace libtensor {
namespace bispace_expr {


template<size_t N1, size_t N2, typename C1, typename C2>
class asym {
public:
	//!	Left expression type
	typedef expr<N1, C1> expr1_t;

	//!	Right expression type
	typedef expr<N2, C2> expr2_t;

private:
	expr1_t m_expr1;
	expr2_t m_expr2;

public:
	asym(const expr1_t &expr1, const expr2_t &expr2) :
		m_expr1(expr1), m_expr2(expr2) { }
	asym(const asym<N1, N2, C1, C2> &s) :
		m_expr1(s.m_expr1), m_expr2(s.m_expr2) { }

	bool equals(const asym<N1, N2, C1, C2> &other) const {
		return m_expr1.equals(other.m_expr1) &&
			m_expr2.equals(other.m_expr2);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		if(i < N1) {
			m_expr1.mark_sym(i, msk, offs);
		} else {
			m_expr2.mark_sym(i - N1, msk, offs + N1);
		}
	}

};

} // namespace bispace_expr
} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_ASYM_H
