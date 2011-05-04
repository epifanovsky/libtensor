#ifndef LIBTENSOR_KERN_MUL_IJKL_PLIQ_PJQK_H
#define LIBTENSOR_KERN_MUL_IJKL_PLIQ_PJQK_H

#include "kern_mul_ijk_pkq_piqj.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ijkl_pliq_pjqk : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_nl, m_np, m_nq;
	size_t m_spa, m_sla, m_sia, m_spb, m_sjb, m_sqb, m_sic, m_sjc, m_skc;

public:
	virtual ~kern_mul_ijkl_pliq_pjqk() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ijk_pkq_piqj &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKL_PLIQ_PJQK_H
