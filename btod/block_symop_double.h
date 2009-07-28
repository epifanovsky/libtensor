#ifndef LIBTENSOR_BLOCK_SYMOP_DOUBLE_H
#define LIBTENSOR_BLOCK_SYMOP_DOUBLE_H

#include "defs.h"
#include "exception.h"
#include "core/block_iterator.h"
#include "core/permutation.h"

namespace libtensor {

template<size_t N>
struct block_symop<N, double> {
public:
	double m_coeff;
	permutation<N> m_perm;

public:
	block_symop() : m_coeff(1.0) { };
};

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_SYMOP_DOUBLE_H
