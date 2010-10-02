#ifndef LIBTENSOR_KERN_MUL_GENERIC_H
#define LIBTENSOR_KERN_MUL_GENERIC_H

#include "kernel_base.h"

namespace libtensor {


class kern_mul_generic : public kernel_base<2, 1> {
public:
	static const char *k_name; //!< Kernel name

private:
	double m_d;

public:
	virtual ~kern_mul_generic() { }

	virtual const char *get_name() const {
		return k_name;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_MUL_GENERIC_H
