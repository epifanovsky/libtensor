#ifndef LIBTENSOR_KERN_MUL_IJ_PIQ_PJQ_H
#define LIBTENSOR_KERN_MUL_IJ_PIQ_PJQ_H

#include "kern_mul_ij_ip_jp.h"

namespace libtensor {


class kern_mul_ij_piq_pjq : public kernel_base<2, 1> {
	friend class kern_mul_ijk_piq_jpkq;
	friend class kern_mul_ijk_piq_pjkq;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_np, m_nq;
	size_t m_spa, m_sia, m_spb, m_sjb, m_sic;

public:
	virtual ~kern_mul_ij_piq_pjq() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_ip_jp &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_PIQ_PJQ_H
