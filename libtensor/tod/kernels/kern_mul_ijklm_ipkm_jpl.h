#ifndef LIBTENSOR_KERN_MUL_IJKLM_IPKM_JPL_H
#define LIBTENSOR_KERN_MUL_IJKLM_IPKM_JPL_H

#include "kern_mul_ijkl_ipl_jpk.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ijklm_ipkm_jpl : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_nl, m_nm, m_np;
	size_t m_sia, m_spa, m_ska, m_sjb, m_spb, m_sic, m_sjc, m_skc, m_slc;

public:
	virtual ~kern_mul_ijklm_ipkm_jpl() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ijkl_ipl_jpk &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKLM_IPKM_JPL_H
