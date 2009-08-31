#ifndef LIBTENSOR_BISPACE_EXPR_EXPR_H
#define LIBTENSOR_BISPACE_EXPR_EXPR_H

#include "defs.h"
#include "exception.h"
#include "../bispace.h"
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

	//!	Number of subexpressions
	static const size_t k_nsubexpr = core_t::k_nsubexpr;

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

	// Returns the number of times this expression contains a subexpression
	template<size_t M, typename D>
	size_t contains(const expr<M, D> &subexpr) const {
		return m_core.contains(subexpr);
	}

	// Returns the location where the subexpression is found
	template<size_t M, typename D>
	size_t locate(const expr<M, D> &subexpr) const {
		return m_core.locate(subexpr);
	}

	template<size_t M, typename D>
	size_t locate_and_permute(const expr<M, D> &other, size_t subexpr,
		size_t start1, size_t start2, permutation<M> &perm) {

		if(subexpr == 0) {
			size_t n = other.contains(*this);
			if(n == 0) {
				// not found
			}
			if(n > 1) {
				// located more than once
			}
			size_t loc = other.locate(*this);
			// permute here
		} else {
			m_core.locate_and_permute(
				other, subexpr, start1, start2, perm);
		}
	}

	template<typename D>
	void build_permutation(const expr<N, D> &other, permutation<N> &perm) {
		for(size_t i = 0; i < k_nsubexpr; i++) {
			locate_and_permute(other, 0, 0, perm);
		}
	}

	const bispace<1> &at(size_t i) const {
		return m_core.at(i);
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

