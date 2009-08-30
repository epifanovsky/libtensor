#ifndef LIBTENSOR_BISPACE_EXPR_EXPR_H
#define LIBTENSOR_BISPACE_EXPR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "../expr_exception.h"

namespace libtensor {
namespace bispace_expr {


/**	\brief Block %index space expression
	\tparam N Expression order.
	\tparam C Expression core type.

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, typename C>
class expr {
public:
	//!	Expression core type
	typedef C core_t;

private:
	core_t m_core;

public:
	expr(const core_t &core) : m_core(core) { }
	expr(const expr<N, C> &e) : m_core(e.m_core) { }

	const core_t &get_core() const {
		return m_core;
	}

	bool equals(const expr<N, C> &other) const {
		return m_core.equals(other.m_core);
	}

	void mark_sym(size_t i, mask<N> &msk) const {
		m_core.mark_sym(i, msk, 0);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		m_core.mark_sym(i, msk, offs);
	}

};


} // namespace bispace_expr
} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_EXPR_H

