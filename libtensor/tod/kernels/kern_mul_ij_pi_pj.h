#ifndef LIBTENSOR_KERN_MUL_IJ_PI_PJ_H
#define LIBTENSOR_KERN_MUL_IJ_PI_PJ_H

#include "kern_mul_i_p_pi.h"
#include "kern_mul_i_pi_p.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ij_pi_pj : public kernel_base<2, 1> {
	friend class kern_mul_ijk_pj_ipk;
	friend class kern_mul_ijk_pj_pik;
	friend class kern_mul_ijk_pji_pk;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_np;
	size_t m_spa, m_spb, m_sic;

public:
	virtual ~kern_mul_ij_pi_pj() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_i_p_pi &z,
		list_t &in, list_t &out);
	static kernel_base<2, 1> *match(const kern_mul_i_pi_p &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJ_PI_PJ_H
