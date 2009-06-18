#ifndef LIBTENSOR_LABELED_BTENSOR_IMPL_H
#define LIBTENSOR_LABELED_BTENSOR_IMPL_H

#include "defs.h"
#include "exception.h"
#include "labeled_btensor.h"
#include "labeled_btensor_expr.h"

namespace libtensor {

template<size_t N, typename T, bool Assignable, typename Label>
inline labeled_btensor<N, T, Assignable, Label>::labeled_btensor(
	btensor_i<N, T> &bt, const letter_expr<N, Label> label) :
	m_bt(bt), m_label(label) {
}

template<size_t N, typename T, bool Assignable, typename Label>
inline btensor_i<N, T>&
labeled_btensor<N, T, Assignable, Label>::get_btensor() const {
	return m_bt;
}

template<size_t N, typename T, bool Assignable, typename Label>
inline bool labeled_btensor<N, T, Assignable, Label>::contains(
	const letter &let) const {
	return m_label.contains(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline size_t labeled_btensor<N, T, Assignable, Label>::index_of(
	const letter &let) const throw(exception) {
	return m_label.index_of(let);
}

template<size_t N, typename T, bool Assignable, typename Label>
inline const letter &labeled_btensor<N, T, Assignable, Label>::letter_at(
	size_t i) const throw(exception) {
	return m_label.letter_at(i);
}

template<size_t N, typename T, typename Label>
inline labeled_btensor<N, T, true, Label>::labeled_btensor(
	btensor_i<N, T> &bt, const letter_expr<N, Label> label) :
	m_bt(bt), m_label(label) {
}

template<size_t N, typename T, typename Label>
inline btensor_i<N, T>&
labeled_btensor<N, T, true, Label>::get_btensor() const {
	return m_bt;
}

template<size_t N, typename T, typename Label>
inline bool labeled_btensor<N, T, true, Label>::contains(
	const letter &let) const {
	return m_label.contains(let);
}

template<size_t N, typename T, typename Label>
inline size_t labeled_btensor<N, T, true, Label>::index_of(
	const letter &let) const throw(exception) {
	return m_label.index_of(let);
}

template<size_t N, typename T, typename Label>
inline const letter &labeled_btensor<N, T, true, Label>::letter_at(
	size_t i) const throw(exception) {
	return m_label.letter_at(i);
}

template<size_t N, typename T, typename Label> template<typename Expr>
labeled_btensor<N, T, true, Label> labeled_btensor<N, T, true, Label>::operator=(
	const labeled_btensor_expr<N, T, Expr> &rhs) throw(exception) {
	/*
	for(size_t i = 0; i < N; i++) if(!expr.contains(letter_at(i))) {
		throw_exc("labeled_btensor<N, T, true, Label>",
			"operator=(const labeled_btensor_expr<N, T, Expr>&)",
			"Index not found in the expression");
	}*/
	rhs.eval(*this);
	return *this;
}

template<size_t N, typename T, typename Label>
template<bool AssignableR, typename LabelR>
labeled_btensor<N, T, true, Label> labeled_btensor<N, T, true, Label>::operator=(
	labeled_btensor<N, T, AssignableR, LabelR> rhs) throw(exception) {
	typedef labeled_btensor_expr_ident<N, T, AssignableR, LabelR> id_t;
	typedef labeled_btensor_expr<N, T, id_t> expr_t;
	expr_t expr(id_t(rhs));
	//expr.assign_to(*this);
	return *this;
}

template<size_t N, typename T, typename Label>
labeled_btensor<N, T, true, Label> labeled_btensor<N, T, true, Label>::operator=(
	labeled_btensor<N, T, true, Label> rhs) throw(exception) {
	typedef labeled_btensor_expr_ident<N, T, true, Label> id_t;
	typedef labeled_btensor_expr<N, T, id_t> expr_t;
	expr_t expr(id_t(rhs));
	//expr.assign_to(*this);
	return *this;
}

} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_IMPL_H
