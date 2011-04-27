#ifndef LIBTENSOR_LOOP_LIST_ELEM1_H
#define LIBTENSOR_LOOP_LIST_ELEM1_H

#include "loop_list_base.h"

namespace libtensor {


/**	\brief Operates nested loops on two arrays with element-wise multiplication
		or division as the kernel.

	\ingroup libtensor_tod
 **/
class loop_list_elem1 : public loop_list_base<1, 1, loop_list_elem1> {
public:
	static const char *k_clazz; //!< Class name

private:
	//!	a_i (+)= k * a_i {*,/} b_i
	struct {
		double m_k;
		size_t m_n;
		size_t m_stepb;
	} m_op;

protected:
	void run_loop(list_t &loop, registers &r, double c,
			bool doadd, bool recip);

private:
	void fn_mult_add(registers &r) const;
	void fn_mult_put(registers &r) const;
	void fn_div_add(registers &r) const;
	void fn_div_put(registers &r) const;
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_MUL1EL_H
