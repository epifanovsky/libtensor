#ifndef LIBTENSOR_LOOP_LIST_ADD_H
#define LIBTENSOR_LOOP_LIST_ADD_H

#include <list>
#include "loop_list_base.h"

namespace libtensor {


/**	\brief Operates nested loops on two arrays with addition and assignment
		as the kernel (b += a)

	\ingroup libtensor_tod
 **/
class loop_list_add : public loop_list_base<1, 1, loop_list_add> {
public:
	static const char *k_clazz; //!< Class name

private:
	//!	b_i += k * a_i
	struct {
		double m_k;
		size_t m_n;
		size_t m_stepa;
		size_t m_stepb;
	} m_daxpy;

	//!	b_ij += k * a_ji
	struct {
		double m_k;
		size_t m_ni;
		size_t m_nj;
		size_t m_stepa;
		size_t m_stepb;
	} m_daxpby_trp;

protected:
	void run_loop(list_t &loop, registers &r, double c);

private:
	void match_l1(list_t &loop, double c);
	void match_l2_a(list_t &loop, double c, size_t w1);
	void match_l2_b(list_t &loop, double c, size_t w1);
	void fn_daxpy(registers &r) const;
	void fn_daxpby_trp(registers &r) const;
};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_ADD_H
