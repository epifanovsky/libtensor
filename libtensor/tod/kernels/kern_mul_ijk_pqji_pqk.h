#ifndef LIBTENSOR_KERN_MUL_IJK_PQJI_PQK_H
#define LIBTENSOR_KERN_MUL_IJK_PQJI_PQK_H

#include "kern_mul_ijk_pji_pk.h"

namespace libtensor {


class kern_mul_ijk_pqji_pqk : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_ni, m_nj, m_nk, m_np, m_nq;
	size_t m_spa, m_sqa, m_sja, m_spb, m_sqb, m_sic, m_sjc;

public:
	virtual ~kern_mul_ijk_pqji_pqk() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_ijk_pji_pk &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_IJK_PQJI_PQK_H
