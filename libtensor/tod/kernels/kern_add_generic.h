#ifndef LIBTENSOR_KERN_ADD_GENERIC_H
#define LIBTENSOR_KERN_ADD_GENERIC_H

#include "kernel_base.h"

namespace libtensor {


/** \brief Generic kernel for additions

 	\ingroup libtensor_tod_kernel
 **/
class kern_add_generic : public kernel_base<2, 1> {
	friend class kern_add_i_i_x_x;
	friend class kern_add_i_x_i_x;

public:
	static const char *k_clazz; //!< Kernel name

private:
	double m_d;

public:
	virtual ~kern_add_generic() { }

	virtual const char *get_name() const {
		return k_clazz;
	}

	virtual void run(const loop_registers<2, 1> &r);

	static kernel_base<2, 1> *match(double d, list_t &in, list_t &out);

};


} // namespace libtensor

#endif // LIBTENSOR_KERN_ADD_GENERIC_H
