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

	bool is_same(const ident<N> &other) const {
		return &m_bis == &other.m_bis;
	}

	template<size_t M, typename D>
	size_t contains(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::contains(*this, subexpr);
	}

	template<size_t M, typename D>
	size_t locate(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::locate(*this, subexpr);
	}

	template<size_t M, typename D>
	void record_pos(const expr<M, D> &supexpr, size_t pos_here,
		size_t (&perm)[M]) const {

		expr< N, ident<N> > subexpr(*this);
		size_t n = supexpr.contains(subexpr);
		if(n == 0) {
			throw expr_exception("libtensor::bispace_expr",
				"ident<N>",
				"locate_and_permute()", __FILE__, __LINE__,
				"Subexpression cannot be located.");
		}
		if(n > 1) {
			throw expr_exception("libtensor::bispace_expr",
				"ident<N>",
				"locate_and_permute()", __FILE__, __LINE__,
				"More than one instance of the subexpression"
				" is found.");
		}
		size_t pos_there = supexpr.locate(subexpr);
		for(size_t i = 0; i < N; i++)
			perm[pos_here + i] = pos_there + i;
	}

	const bispace<1> &at(size_t i) const {
		return m_bis.at(i);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		const mask<N> &symmsk = m_bis.get_sym_mask(i);
		for(size_t j = 0; j < N; j++)
			msk[offs + j] = msk[offs + j] || symmsk[j];
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
