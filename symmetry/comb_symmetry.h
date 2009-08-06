#ifndef LIBTENSOR_COMB_SYMMETRY_H
#define LIBTENSOR_COMB_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "core/permutation.h"
#include "symmetry_base.h"

namespace libtensor {

template<size_t N, typename T>
class comb_symmetry_base :
	public symmetry_base< N, T, comb_symmetry_base<N, T> > {
public:
	virtual comb_symmetry_base<N, T> *clone() const = 0;
};

/**	\brief Combination of two symmetry classes

	\ingroup libtensor
 **/
template<size_t N, size_t M, typename T>
class comb_symmetry : public comb_symmetry_base<N + M, T> {
private:
	symmetry_i<N, T> &m_sym1;
	symmetry_i<M, T> &m_sym2;
	permutation<N + M> m_perm;

public:
	comb_symmetry(symmetry_i<N, T> &sym1, symmetry_i<M, T> &sym2,
		permutation<N + M> &perm);
	virtual comb_symmetry_base<N + M, T> *clone() const;
};

template<size_t N, size_t M, typename T>
comb_symmetry<N, M, T>::comb_symmetry(
	symmetry_i<N, T> &sym1, symmetry_i<M, T> &sym2,
	permutation<N + M> &perm)
: m_sym1(sym1), m_sym2(sym2), m_perm(perm) {

}

template<size_t N, size_t M, typename T>
comb_symmetry_base<N + M, T> *comb_symmetry<N, M, T>::clone() const {

	return new comb_symmetry<N, M, T>(m_sym1, m_sym2, m_perm);
}


} // namespace libtensor

#endif // LIBTENSOR_COMB_SYMMETRY_H
