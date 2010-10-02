#include "kern_mul_generic.h"

namespace libtensor {


const char *kern_mul_generic::k_name = "kern_generic";


void kern_mul_generic::run(const loop_registers<2, 1> &r) {

	r.m_ptrb[0][0] += r.m_ptra[0][0] * r.m_ptra[1][0] * m_d;

}


kernel_base<2, 1> *kern_mul_generic::match(double d, list_t &in, list_t &out) {

	kernel_base<2, 1> *kern = 0;

	{
		kern_mul_generic k;
		k.m_d = d;
		kern = new kern_mul_generic(k);
	}
	return kern;
}


} // namespace libtensor
