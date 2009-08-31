#ifndef LIBTENSOR_BISPACE_EXPR_IDENT_H
#define LIBTENSOR_BISPACE_EXPR_IDENT_H

#include "defs.h"
#include "exception.h"
#include "../expr_exception.h"
#include "expr.h"

namespace libtensor {
namespace bispace_expr {


template<size_t N>
class ident {
public:
	//!	Number of subexpressions
	static const size_t k_nsubexpr = 1;

private:
	template<size_t M, typename D, int Dummy = 0>
	struct subexpr_functor {

		static size_t contains(
			const ident<N> &expr, const expr<M, D> &subexpr) {
			return 0;
		}

		static size_t locate(
			const ident<N> &expr, const expr<M, D> &subexpr) {
			throw expr_exception("libtensor::bispace_expr",
				"ident<N>::subexpr_functor<M, D, 0>",
				"locate()", __FILE__, __LINE__,
				"Subexpression cannot be located.");
		}
	};

private:
	const bispace<N> &m_bis;

public:
	ident(const bispace<N> &bis) : m_bis(bis) { }
	ident(const ident<N> &id) : m_bis(id.m_bis) { }

	bool equals(const ident<N> &other) const {
		return m_bis.equals(other.m_bis);
	}

	template<size_t M, typename D>
	size_t contains(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::contains(*this, subexpr);
	}

	template<size_t M, typename D>
	size_t locate(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::locate(*this, subexpr);
	}

	bool is_same(const ident<N> &other) const {
		return &m_bis == &other.m_bis;
	}

	const bispace<1> &at(size_t i) const {
		return m_bis.at(i);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		msk[offs + i] = true;
	}

};


template<size_t N> template<int Dummy>
struct ident<N>::subexpr_functor<N, ident<N>, Dummy> {

	static size_t contains(
		const ident<N> &expr, const expr< N, ident<N> > &subexpr) {

		return expr.is_same(subexpr.get_core()) ? 1 : 0;
	}

	static size_t locate(
		const ident<N> &expr, const expr< N, ident<N> > &subexpr) {

		if(!expr.is_same(subexpr.get_core())) {
			throw expr_exception("libtensor::bispace_expr",
				"ident<N>::subexpr_functor<N, ident<N>, 0>",
				"locate()", __FILE__, __LINE__,
				"Subexpression cannot be located.");
		}
		return 0;
	}
};


} // namespace bispace_expr
} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_IDENT_H
