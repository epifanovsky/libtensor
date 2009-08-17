#ifndef LIBTENSOR_TRANSF_DOUBLE_H
#define LIBTENSOR_TRANSF_DOUBLE_H

#include "defs.h"
#include "exception.h"
#include "core/permutation.h"
#include "core/transf.h"

namespace libtensor {

template<size_t N>
struct transf<N, double> {
public:
	double m_coeff;
	permutation<N> m_perm;

public:
	transf() : m_coeff(1.0) { }
	transf(const transf<N, double> &tr) : m_coeff(tr.m_coeff), m_perm(tr.m_perm) { }
	void reset() { m_coeff = 1.0; m_perm.reset(); }
	void permute(const permutation<N> &perm) { m_perm.permute(perm); }
	void multiply(double c) { m_coeff *= c; }
	void transform(const transf<N, double> &tr) {
		m_coeff *= tr.m_coeff;
		m_perm.permute(tr.m_perm);
	}
	void invert() {
		m_coeff = (m_coeff == 0.0 ? 0.0 : 1.0/m_coeff);
		m_perm.invert();
	}
};

} // namespace libtensor

#endif // LIBTENSOR_TRANSF_DOUBLE_H
