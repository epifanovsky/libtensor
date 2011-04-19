#ifndef LIBTENSOR_KERN_MUL_IJ_JP_IP_H
#define LIBTENSOR_KERN_MUL_IJ_JP_IP_H

#include "kern_mul_i_ip_p.h"

namespace libtensor {


class kern_mul_ij_jp_ip : public kernel_base<2, 1> {
	friend class kern_mul_ij_pjq_piq;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_np;
	size_t m_sja, m_sib, m_sic;

public:
	virtual ~kern_mul_ij_jp_ip() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_i_ip_p &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_JP_IP_H
