#ifndef LIBTENSOR_KERN_MUL_IJKL_PLJQ_IPQK_H
#define LIBTENSOR_KERN_MUL_IJKL_PLJQ_IPQK_H

#include "kern_mul_ijk_pkq_ipqj.h"

namespace libtensor {


/**
	\ingroup libtensor_tod_kernel
 **/
class kern_mul_ijkl_pljq_ipqk : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_nl, m_np, m_nq;
	size_t m_spa, m_sla, m_sja, m_sib, m_spb, m_sqb, m_sic, m_sjc, m_skc;

public:
	virtual ~kern_mul_ijkl_pljq_ipqk() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ijk_pkq_ipqj &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJKL_PLJQ_IPQK_H
