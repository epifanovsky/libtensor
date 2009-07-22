#ifndef LIBTENSOR_COMB_SYMMETRY_H
#define LIBTENSOR_COMB_SYMMETRY_H

#include "defs.h"
#include "exception.h"
#include "permutation.h"
#include "symmetry_base.h"

namespace libtensor {

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
};

template<size_t N, size_t M, typename T>
comb_symmetry::comb_symmetry(symmetry_i<N, T> &sym1, symmetry_i<M, T> &sym2,
		permutation<N + M> &perm)
: m_sym1(sym1), m_sym2(sym2), m_perm(perm) {

}

} // namespace libtensor

#endif // LIBTENSOR_COMB_SYMMETRY_H
