#ifndef LIBTENSOR_TRANSF_DOUBLE_H
#define LIBTENSOR_TRANSF_DOUBLE_H

#include "../defs.h"
#include "../exception.h"
#include "../core/permutation.h"
#include "../core/transf.h"

namespace libtensor {

template<size_t N>
class transf<N, double> {
private:
	double m_coeff;
	permutation<N> m_perm;

public:
	transf() : m_coeff(1.0) { }
	transf(const transf<N, double> &tr) : m_coeff(tr.m_coeff), m_perm(tr.m_perm) { }
	void reset() { m_coeff = 1.0; m_perm.reset(); }
	void permute(const permutation<N> &perm) { m_perm.permute(perm); }
	void scale(double c) { m_coeff *= c; }
	void transform(const transf<N, double> &tr) {
		m_coeff *= tr.m_coeff;
		m_perm.permute(tr.m_perm);
	}
	void apply(index<N> &idx) const {
		idx.permute(m_perm);
	}
	void invert() {
		m_coeff = (m_coeff == 0.0 ? 0.0 : 1.0/m_coeff);
		m_perm.invert();
	}
	bool is_identity() const { return m_coeff == 1.0 && m_perm.is_identity(); }

	//! member access functions
	//@{
	/** \brief returns the coefficient
	 **/
	double& get_coeff() { return m_coeff; }

	/** \brief returns the coefficient
	 **/
	const double& get_coeff() const { return m_coeff; }

	/** \brief returns the permutation
	 **/
	permutation<N>& get_perm() { return m_perm; }

	/** \brief returns the permutation
	 **/
	const permutation<N>& get_perm() const { return m_perm; }
	//@}

	//! Comparison operators
	//@{
	/** \brief equal comparison
	 **/
	bool operator==( const transf<N,double>& tr ) const {
		return ( (m_coeff==tr.m_coeff) && (m_perm==tr.m_perm) );
	}

	/** \brief unequal comparison
	 **/
	bool operator!=( const transf<N,double>& tr ) const
	{	return (!operator==(tr)); }
	//@}
};

} // namespace libtensor

#endif // LIBTENSOR_TRANSF_DOUBLE_H
