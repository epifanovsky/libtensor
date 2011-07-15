#ifndef LIBTENSOR_KERN_MUL_IJ_JIPQ_QP_H
#define LIBTENSOR_KERN_MUL_IJ_JIPQ_QP_H

#include "kern_mul_i_ipq_qp.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ij_jipq_qp : public kernel_base<2, 1> {
	friend class kern_mul_ijk_kjpq_iqp;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_np, m_nq;
	size_t m_sja, m_sia, m_spa, m_sqb, m_sic;

public:
	virtual ~kern_mul_ij_jipq_qp() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_i_ipq_qp &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_JIPQ_QP_H
