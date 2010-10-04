#ifndef LIBTENSOR_KERN_MUL_X_P_P_H
#define LIBTENSOR_KERN_MUL_X_P_P_H

#include "kern_mul_generic.h"

namespace libtensor {


class kern_mul_x_p_p : public kernel_base<2, 1> {
public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;
	size_t m_np;
	size_t m_spa, m_spb;

public:
	virtual ~kern_mul_x_p_p() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(const kern_mul_generic &z,
		list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_X_P_P_H
