#ifndef LIBTENSOR_BISPACE_EXPR_SYM_H
#define LIBTENSOR_BISPACE_EXPR_SYM_H

#include "expr.h"

namespace libtensor {
namespace bispace_expr {


/**	\brief Expression core indicating a symmetry relationship between
		subspaces
	\tparam N Order of symmetric expression.
	\tparam K Number of symmetric parts.
	\tparam C Symmetric expression core.

	\ingroup libtensor_bispace_expr
 **/
template<size_t N, size_t K, typename C>
class sym {
public:
	//!	Left expression type
	typedef expr< N * (K - 1), sym<N, K - 1, C> > expr1_t;

	//!	Right expression type
	typedef expr< N, sym<N, 1, C> > expr2_t;

	//!	Number of subexpressions
	static const size_t k_nsubexpr =
		expr1_t::k_nsubexpr + expr2_t::k_nsubexpr;

private:
	template<size_t M, typename D, int Dummy = 0>
	struct subexpr_functor {

		static size_t contains(
			const sym<N, K, C> &expr,
			const expr<M, D> &subexpr) {

			size_t n = 0;
			for(size_t i = 0; i < K; i++) {
				n += expr.get_subexpr(i).contains(subexpr);
			}
			return n;
		}

		static size_t locate(
			const sym<N, K, C> &expr,
			const expr<M, D> &subexpr) {

			size_t pos = 0;
			for(size_t i = 0; i < K; i++) {
				if(expr.get_subexpr(i).contains(subexpr)) {
					return pos + expr.get_subexpr(i).
						locate(subexpr);
				}
				pos += N;
			}
			throw expr_exception("libtensor::bispace_expr",
				"sym<N, K, C>::subexpr_functor<M, D, 0>",
				"locate()", __FILE__, __LINE__,
				"Subexpression cannot be located.");
		}
	};

private:
	expr1_t m_expr1;
	expr2_t m_expr2;

public:
	sym(const expr1_t &expr1, const expr2_t &expr2) :
		m_expr1(expr1), m_expr2(expr2) {

		if(!get_subexpr(0).equals(m_expr2.get_core())) {
			throw expr_exception("libtensor::bispace_expr",
				"sym<N, K, C>", "sym()", __FILE__, __LINE__,
				"Incompatible subspaces.");
		}
	}
	sym(const sym<N, K, C> &s) : m_expr1(s.m_expr1), m_expr2(s.m_expr2) { }

	const sym<N, 1, C> &get_first() const {
		return m_expr1.get_core().get_first();
	}

	const sym<N, 1, C> &get_subexpr(size_t i) const {
		if(i >= K) {
			throw out_of_bounds("libtensor::bispace_expr",
				"sym<N, K, C>", "get_subexpr(size_t)",
				__FILE__, __LINE__,
				"Subexpression index is out of bounds.");
		}
		if(i == K - 1) {
			return m_expr2.get_core();
		} else {
			return m_expr1.get_core().get_subexpr(i);
		}
	}

	bool equals(const sym<N, K, C> &other) const {
		return get_subexpr(0).equals(other.get_subexpr(0));
	}

	template<size_t M, typename D>
	size_t contains(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::contains(*this, subexpr);
	}

	template<size_t M, typename D>
	size_t locate(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::locate(*this, subexpr);
	}

	const bispace<1> &at(size_t i) const {
		return i < N * (K - 1) ?
			m_expr1.at(i) : m_expr2.at(i - N * (K - 1));
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		size_t imin = i % N;
		for(size_t j = 0; j < K - 1; j++) {
			m_expr1.mark_sym(imin, msk, offs + j * N);
		}
		m_expr2.mark_sym(imin, msk, offs + (K - 1) * N);
	}

};


template<size_t N, typename C>
class sym<N, 1, C> {
public:
	//!	Symmetric expression type
	typedef expr<N, C> expr_t;

private:
	template<size_t M, typename D, int Dummy = 0>
	struct subexpr_functor {

		static size_t contains(
			const sym<N, 1, C> &expr,
			const expr<M, D> &subexpr) {

			return expr.get_expr().contains(subexpr);
		}

		static size_t locate(
			const sym<N, 1, C> &expr,
			const expr<M, D> &subexpr) {

			return expr.get_expr().locate(subexpr);
		}
	};

private:
	expr_t m_expr;

public:
	sym(const expr_t expr) : m_expr(expr) { }
	sym(const sym<N, 1, C> &s) : m_expr(s.m_expr) { }

	const expr_t &get_expr() const {
		return m_expr;
	}

	const sym<N, 1, C> &get_subexpr(size_t i) const {
		if(i != 0) {
			throw out_of_bounds("libtensor::bispace_expr",
				"sym<N, 1, C>", "get_subexpr(size_t)",
				__FILE__, __LINE__,
				"Subexpression index is out of bounds.");
		}
		return *this;
	}

	bool equals(const sym<N, 1, C> &other) const {
		return m_expr.equals(other.m_expr);
	}

	template<size_t M, typename D>
	size_t contains(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::contains(*this, subexpr);
	}

	template<size_t M, typename D>
	size_t locate(const expr<M, D> &subexpr) const {
		return subexpr_functor<M, D>::locate(*this, subexpr);
	}

	const bispace<1> &at(size_t i) const {
		return m_expr.at(i);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		m_expr.mark_sym(i, msk, offs);
	}

};


} // namespace bispace_expr
} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_SYM_H
