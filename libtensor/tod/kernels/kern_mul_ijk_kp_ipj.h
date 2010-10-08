#ifndef LIBTENSOR_KERN_MUL_IJK_KP_IPJ_H
#define LIBTENSOR_KERN_MUL_IJK_KP_IPJ_H

#include "kern_mul_ij_jp_pi.h"

namespace libtensor {


class kern_mul_ijk_kp_ipj : public kernel_base<2, 1> {
	friend class kern_mul_ijk_pkq_ipqj;
	friend class kern_mul_ijk_pkq_piqj;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np;
	size_t m_ska, m_sib, m_spb, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_kp_ipj() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_jp_pi &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_KP_IPJ_H
