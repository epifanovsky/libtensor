#ifndef LIBTENSOR_BISPACE_EXPR_IDENT_H
#define LIBTENSOR_BISPACE_EXPR_IDENT_H

namespace libtensor {
namespace bispace_expr {


template<size_t N>
class ident {
private:
	const bispace<N> &m_bis;

public:
	ident(const bispace<N> &bis) : m_bis(bis) { }
	ident(const ident<N> &id) : m_bis(id.m_bis) { }

	bool equals(const ident<N> &other) const {
		return m_bis.equals(other.m_bis);
	}

	template<size_t M>
	void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
		msk[offs + i] = true;
	}

};


} // namespace bispace_expr
} // namespace libtensor

#endif // LIBTENSOR_BISPACE_EXPR_IDENT_H
