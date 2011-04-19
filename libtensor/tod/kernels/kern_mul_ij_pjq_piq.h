#ifndef LIBTENSOR_KERN_MUL_IJ_PJQ_PIQ_H
#define LIBTENSOR_KERN_MUL_IJ_PJQ_PIQ_H

#include "kern_mul_ij_jp_ip.h"

namespace libtensor {


class kern_mul_ij_pjq_piq : public kernel_base<2, 1> {
	friend class kern_mul_ijk_pkiq_pjq;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_np, m_nq;
	size_t m_sja, m_spa, m_sib, m_spb, m_sic;

public:
	virtual ~kern_mul_ij_pjq_piq() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_jp_ip &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_PJQ_PIQ_H
