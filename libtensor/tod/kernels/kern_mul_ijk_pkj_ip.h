#ifndef LIBTENSOR_KERN_MUL_IJK_PKJ_IP_H
#define LIBTENSOR_KERN_MUL_IJK_PKJ_IP_H

#include "kern_mul_ij_pji_p.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pkj_ip : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np;
	size_t m_spa, m_ska, m_sib, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_pkj_ip() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_pji_p &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PKJ_IP_H
