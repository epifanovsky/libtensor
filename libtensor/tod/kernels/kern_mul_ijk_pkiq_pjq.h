#ifndef LIBTENSOR_KERN_MUL_IJK_PKIQ_PJQ_H
#define LIBTENSOR_KERN_MUL_IJK_PKIQ_PJQ_H

#include "kern_mul_ij_pjq_piq.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ijk_pkiq_pjq : public kernel_base<2, 1> {
	friend class kern_mul_ijkl_pliq_jpkq;
	friend class kern_mul_ijkl_pljq_ipkq;
	friend class kern_mul_ijkl_pljq_pikq;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np, m_nq;
	size_t m_spa, m_ska, m_sia, m_spb, m_sjb, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_pkiq_pjq() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ij_pjq_piq &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PKIQ_PJQ_H
