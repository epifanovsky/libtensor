#ifndef LIBTENSOR_KERN_MUL_IJK_PIQ_PJQK_H
#define LIBTENSOR_KERN_MUL_IJK_PIQ_PJQK_H

#include "kern_mul_ijk_ip_jpk.h"

namespace libtensor {


class kern_mul_ijk_piq_pjqk : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np, m_nq;
	size_t m_spa, m_sia, m_spb, m_sjb, m_sqb, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_piq_pjqk() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ijk_ip_jpk &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PIQ_PJQK_H
