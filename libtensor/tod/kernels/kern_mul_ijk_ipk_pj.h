#ifndef LIBTENSOR_KERN_MUL_IJK_IPK_PJ_H
#define LIBTENSOR_KERN_MUL_IJK_IPK_PJ_H

#include "kern_mul_ij_pj_pi.h"

namespace libtensor {


class kern_mul_ijk_ipk_pj : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np;
	size_t m_sia, m_spa, m_spb, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_ipk_pj() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_pj_pi &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_IPK_PJ_H
